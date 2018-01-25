#include "IGPUTransientBuffer.h"
#include "COpenGLPersistentlyMappedBuffer.h"
#include "os.h"
#include "FW_Mutex.h"
#include <sstream>

using namespace irr;
using namespace video;


#if defined(_IRR_WINDOWS_API_)
// ----------------------------------------------------------------
// Windows specific functions
// ----------------------------------------------------------------

#ifdef _IRR_XBOX_PLATFORM_
#include <xtl.h>
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <time.h>
#endif

#else

#include <sys/time.h>

#endif // defined



//#define _EXTREME_DEBUG


IGPUTransientBuffer::IGPUTransientBuffer(IVideoDriver* driver, IGPUBuffer* buffer, const bool& growable, const bool& autoFlush, const bool& threadSafe, core::LeakDebugger* dbgr)
                                    :   lastChanged(0), canGrow(growable), flushOnWait(autoFlush), Driver(driver), underlyingBuffer(buffer), underlyingBufferAsMapped(dynamic_cast<IGPUMappedBuffer*>(underlyingBuffer)),
                                        totalTrueFreeSpace(0), totalFreeSpace(0), largestFreeChunkSize(0), trueLargestFreeChunkSize(0), leakTracker(dbgr)
{
    if (leakTracker)
        leakTracker->registerObj(this);

    Allocation first;
    first.state = Allocation::EAS_FREE;
    first.fence = NULL;
    first.start = 0;
    if (underlyingBuffer)
    {
        underlyingBuffer->grab();
        first.end = underlyingBuffer->getSize();
        totalTrueFreeSpace = totalFreeSpace = largestFreeChunkSize = trueLargestFreeChunkSize = underlyingBuffer->getSize();
    }
    else
        first.end = 0;
    allocs.reserve(1024);
    allocs.push_back(first);

    if (threadSafe)
    {
        mutex = new FW_Mutex();
        allocationChanged = new FW_ConditionVariable(mutex);
        allocMutex = new FW_Mutex();
    }
    else
    {
        mutex = NULL;
        allocationChanged = NULL;
        allocMutex = NULL;
    }
}

IGPUTransientBuffer::~IGPUTransientBuffer()
{
    if (leakTracker)
        leakTracker->deregisterObj(this);

    if (mutex)
        mutex->Get();

    if (underlyingBuffer)
        underlyingBuffer->drop();

    for (size_t i=0; i<allocs.size(); i++)
    {
        if (allocs[i].fence)
            allocs[i].fence->drop();
    }
    if (mutex)
    {
        mutex->Release();
        delete mutex;

        if (allocationChanged)
            delete allocationChanged;
    }

    if (allocMutex)
        delete allocMutex;
}
//
bool IGPUTransientBuffer::Validate()
{
    if (mutex)
        mutex->Get();
    bool retval = validate_ALREADYMUTEXED();
    if (mutex)
        mutex->Release();
    return retval;
}

bool IGPUTransientBuffer::validate_ALREADYMUTEXED()
{
    if (!underlyingBuffer)
        return false;

    if (allocs.size()<1)
        return false;

    if (allocs[0].start!=0||allocs.back().end!=underlyingBuffer->getSize())
        return false;

    for (size_t i=0; i<allocs.size(); i++)
    {
        if (invalidState(allocs[i]))
            return false;

        if (allocs[i].end<=allocs[i].start)
            return false;

        if (i>0&&allocs[i-1].end!=allocs[i].start)
            return false;
    }
/*
    size_t tmpSize = (underlyingBuffer->getSize()+7)/8;
    uint8_t* tmpData = new uint8_t[tmpSize];
    memset(tmpData,0,tmpSize);
    for (size_t i=0; i<allocs.size(); i++)
    for (size_t j=allocs[i].start; j<allocs[i].end; j++)
    {
        size_t addr = j/8;
        uint8_t mask = 0x1u<<(j-8*addr);
        uint8_t& location = tmpData[addr];
        if (location&mask)
        {
            delete [] tmpData;
            os::Printer::log("DOUBLY ALLOCATED RANGE!\n",ELL_ERROR);
            return false;
        }
        location |= mask;
    }

    delete [] tmpData;*/

    return true;
}

void IGPUTransientBuffer::PrintDebug(bool needsMutex)
{
    if (needsMutex&&mutex)
        mutex->Get();

    os::Printer::log("==========================GPU TRANSIENT BUFFER INFO==========================\n",ELL_INFORMATION);
    os::Printer::log("==========================          START          ==========================\n",ELL_INFORMATION);
    for (int32_t i=0; i<allocs.size(); i++)
    {
        std::ostringstream infoOut("Block:");
        infoOut.seekp(0,std::ios_base::end);
        infoOut << i << "\t\t\tStart:" << allocs[i].start << "\t\t\tEnd:  " << allocs[i].end << "\t\t\tFence:" << reinterpret_cast<size_t &>(allocs[i].fence) << "\tRefCnt:";
        if (allocs[i].fence)
            infoOut << allocs[i].fence->getReferenceCount();
        else
            infoOut << "0";
        infoOut << "\t\t\tState:" << allocs[i].state << "\n";
        os::Printer::log(infoOut.str().c_str(),ELL_INFORMATION);
    }
    os::Printer::log("==========================           END           ==========================\n",ELL_INFORMATION);

    if (needsMutex&&mutex)
        mutex->Release();
}

inline size_t roundUpStart(size_t index, const size_t& alignment)
{
    index += alignment-1;
    index /= alignment;
    return index*alignment;
}

IGPUTransientBuffer::E_ALLOC_RETURN_STATUS IGPUTransientBuffer::Alloc(size_t &offsetOut, const size_t &maxSize, const size_t& alignment, E_WAIT_POLICY waitPolicy, bool growIfTooSmall)
{
    if (maxSize==0)
        return EARS_FAIL;


    if (!canGrow)
        growIfTooSmall = false;

    if (!allocMutex)
        waitPolicy = EWP_DONT_WAIT;
    //only one thread can spinlock wait to allocate
    //makes sure that if we have to wait for all of the
    //allocated chunks to be unmapped, then it will eventually happen
    if (waitPolicy&&allocMutex)
        allocMutex->Get();

    if (mutex)
        mutex->Get();
    if (!underlyingBuffer||(!canGrow&&maxSize>underlyingBuffer->getSize()))
    {
        if (mutex)
            mutex->Release();
        if (waitPolicy&&allocMutex)
            allocMutex->Release();
        return EARS_ERROR;
    }

#ifdef _EXTREME_DEBUG
    if (!validate_ALREADYMUTEXED())
    {
        os::Printer::log("TRNASIENT BUFFER VALIDATION FAILED!\n",ELL_ERROR);
        PrintDebug(false);
    }
#endif // _EXTREME_DEBUG

    // defragment all the time
    // grow if everything is unmapped
    // circle releasing fence
    do
    {
        bool noFencesToCycle = true;
        bool allFree = true;
        //defragment while checking if all are unmapped
        size_t j=0;
        for (size_t i=1; i<allocs.size(); i++)
        {
            bool retest = false;
            //not the same as next
            if (allocs[j].state!=allocs[i].state||allocs[j].fence!=allocs[i].fence)
            {
                switch (allocs[j].state)
                {
                    case Allocation::EAS_FREE:
                        //only wait on fence if free
                        if (allocs[j].fence)
                        {
                            switch (allocs[j].fence->waitCPU(0,flushOnWait))
                            {
                                case EDFR_TIMEOUT_EXPIRED:
                                    noFencesToCycle = false;
                                    break;
                                default: //any other thing
                                    allocs[j].fence->drop();
                                    allocs[j].fence = NULL;

                                    totalTrueFreeSpace += allocs[i].start-allocs[j].start;

                                    retest = !allocs[i].fence;
                                    break;
                            }
                        }
                        //could have changed and needs merge
                        if (!allocs[j].fence)
                        {
                            //found a block large enough to allocate from
                            size_t startAligned = roundUpStart(allocs[j].start,alignment);
                            if (startAligned<allocs[i].start&&allocs[i].start-startAligned>=maxSize)
                            {
                                offsetOut = startAligned;

                                if (allocs[j].start!=startAligned)
                                {
                                    //front of the allocated space will be freed
                                    allocs[j].state = Allocation::EAS_FREE;
                                    allocs[j++].end = startAligned;
                                    //insert new chunk info straight after
                                    Allocation tmp;
                                    tmp.state = Allocation::EAS_ALLOCATED;
                                    tmp.fence = NULL;
                                    tmp.start = startAligned;
                                    tmp.end = startAligned+maxSize;
                                    //insert new chunk after old
                                    if (j<i)
                                        allocs[j] = tmp;
                                    else
                                    {
                                        allocs.insert(allocs.begin()+j,tmp);
                                        i++;//need to move our next index along
                                    }
                                }
                                else
                                {
                                    //allocate from the start of current chunk
                                    allocs[j].state = Allocation::EAS_ALLOCATED;
                                    allocs[j].end = startAligned+maxSize;
                                }
                                //no j/the current maps to the chunk with EAS_ALLOCATED state
                                if (startAligned+maxSize<allocs[i].start)
                                {
                                    //new free chunk to insert after current chunk
                                    Allocation tmp;
                                    tmp.state = Allocation::EAS_FREE;
                                    tmp.fence = NULL;
                                    tmp.start = startAligned+maxSize;
                                    tmp.end = allocs[i].start;

                                    //if the next chunk desc we look at is FAAAR away
                                    j++;
                                    if (j<i)
                                        allocs[j] = tmp;
                                    else
                                    {
                                        //j = current desc index equals i= next desc index
                                        allocs.insert(allocs.begin()+j,tmp);
                                        i++;
                                    }
                                }

                                if (j+1<i)
                                {
                                    j++;
                                    for (; i<allocs.size(); i++,j++)
                                    {
                                        allocs[j] = allocs[i];
                                    }
                                    allocs.resize(j);
                                }
                                totalTrueFreeSpace -= maxSize;
                                totalFreeSpace -= maxSize;

                                if (mutex)
                                    mutex->Release();
                                if (waitPolicy&&allocMutex)
                                    allocMutex->Release();
                                return EARS_SUCCESS;
                            }
                        }
                        break;
                    case Allocation::EAS_ALLOCATED:
                        allFree = false;
                        break;
                    case Allocation::EAS_PENDING_RENDER_CMD:
                        allFree = false;
                        break;
                }

                if (retest)
                {
                    i--;
                    continue;
                }

                if (j+1<i)
                {
                    allocs[j].end = allocs[i].start;
                    j++;
                    allocs[j] = allocs[i];
                }
                else
                    j++;
            } //will be removed (collapsed) so drop reference to fence
            else if (allocs[i].fence)
                allocs[i].fence->drop();
        }

        //trim and fix the end
        if (j+1<allocs.size())
        {
            allocs[j].end = allocs.back().end;
            allocs.resize(j+1);
        }

        //process last chunk
        if (allocs[j].state==Allocation::EAS_FREE)
        {
            if (allocs[j].fence)
            {
                switch (allocs[j].fence->waitCPU(0,flushOnWait))
                {
                    case EDFR_TIMEOUT_EXPIRED:
                        noFencesToCycle = false;
                        break;
                    default: //any other thing
                        allocs[j].fence->drop();
                        allocs[j].fence = NULL;

                        totalTrueFreeSpace += allocs[j].end-allocs[j].start;
                        break;
                }
            }
            //could have changed
            if (!allocs[j].fence)
            {
                size_t startAligned = roundUpStart(allocs[j].start,alignment);
                if (startAligned<allocs[j].end&&allocs[j].end-startAligned>=maxSize)
                {
                    offsetOut = startAligned;
                    size_t bufferEnd = allocs[j].end;

                    if (allocs[j].start!=startAligned)
                    {
                        //front of the allocated space will be freed
                        allocs[j].state = Allocation::EAS_FREE;
                        allocs[j++].end = startAligned;
                        //insert new chunk info straight after
                        Allocation tmp;
                        tmp.state = Allocation::EAS_ALLOCATED;
                        tmp.fence = NULL;
                        tmp.start = startAligned;
                        tmp.end = startAligned+maxSize;
                        //insert new chunk after old
                        allocs.push_back(tmp);
                    }
                    else
                    {
                        //allocate from the start of current chunk
                        allocs[j].state = Allocation::EAS_ALLOCATED;
                        allocs[j].end = startAligned+maxSize;
                    }

                    if (startAligned+maxSize<bufferEnd)
                    {
                        //new free chunk to insert after current chunk
                        Allocation tmp;
                        tmp.state = Allocation::EAS_FREE;
                        tmp.fence = NULL;
                        tmp.start = startAligned+maxSize;
                        tmp.end = bufferEnd;

                        allocs.push_back(tmp);
                    }

                    totalTrueFreeSpace -= maxSize;
                    totalFreeSpace -= maxSize;

                    if (mutex)
                        mutex->Release();
                    if (waitPolicy&&allocMutex)
                        allocMutex->Release();
                    return EARS_SUCCESS;
                }
            }
        }
        else
            allFree = false;

#ifdef _EXTREME_DEBUG
        if (!validate_ALREADYMUTEXED())
        {
            os::Printer::log("TRNASIENT BUFFER VALIDATION FAILED!\n",ELL_ERROR);
            PrintDebug(false);
        }
#endif // _EXTREME_DEBUG

        if (allFree)
        {
            if (growIfTooSmall)
            {
                //grow and quit
                size_t oldBufferSz = allocs.back().end;
                bool lastChunkFree = allocs.back().state==Allocation::EAS_FREE&&(!allocs.back().fence);
                if (!underlyingBuffer->reallocate(maxSize+(lastChunkFree ? allocs.back().start:oldBufferSz), true, false))
                {
                    underlyingBuffer->drop();
                    underlyingBuffer = NULL;
                    if (mutex)
                        mutex->Release();
                    if (waitPolicy&&allocMutex)
                        allocMutex->Release();
                    return EARS_ERROR;
                }

                if (lastChunkFree)
                {
                    totalTrueFreeSpace -= oldBufferSz-allocs.back().start;
                    totalFreeSpace -= oldBufferSz-allocs.back().start;

                    offsetOut = allocs.back().start;
                    allocs.back().end = allocs.back().start+maxSize;
                    allocs.back().state = Allocation::EAS_ALLOCATED;
                }
                else
                {
                    offsetOut = oldBufferSz;
                    Allocation newLast;
                    newLast.state = Allocation::EAS_ALLOCATED;
                    newLast.fence = NULL;
                    newLast.start = oldBufferSz;
                    newLast.end = oldBufferSz+maxSize;
                    allocs.push_back(newLast);
                }

                if (mutex)
                    mutex->Release();
                if (waitPolicy&&allocMutex)
                    allocMutex->Release();
                return EARS_SUCCESS;
            }
			else if (noFencesToCycle||(!noFencesToCycle&&waitPolicy<EWP_WAIT_FOR_GPU_FREE))
                break;
        }

        //allocationChanged condition not changed when new fence placed
        //we already had too little sequential memory to allocate,
        //fence will make it even less so pointless to examine again
        if (waitPolicy==EWP_WAIT_FOR_CPU_UNMAP||(noFencesToCycle&&waitPolicy==EWP_WAIT_FOR_GPU_FREE))
        {
            size_t oldChanged = lastChanged;
            while (lastChanged==oldChanged)
            {
                allocationChanged->WaitForCondition(mutex);
            }
            //something has changed
        }
        //else if waiting for GPU, keep cycling fences to remove them
    } while (waitPolicy);


    if (mutex)
        mutex->Release();
    if (waitPolicy&&allocMutex)
        allocMutex->Release();
    return EARS_FAIL;
}


bool IGPUTransientBuffer::Commit(const size_t& start, const size_t& end)
{
    if (start>end)
        return false;
    else if (start==end)
        return true;

    if (mutex)
        mutex->Get();
    if (!underlyingBuffer||end>underlyingBuffer->getSize()||start>=underlyingBuffer->getSize())
    {
        if (mutex)
            mutex->Release();
        return false;
    }
    uint32_t index;
    if (!findFirstChunk(index,start)||invalidState(allocs[index]))
    {
        if (mutex)
            mutex->Release();
#ifdef _DEBUG
        //os::Printer::log("INVALID STATE IN IGPUTransientBuffer::commit",ELL_ERROR);
#endif // _DEBUG
        return false;
    }


    if (allocs[index].state==Allocation::EAS_ALLOCATED)
    {
        if (allocs[index].start<start)
        {
            Allocation tmp;
            tmp.state = Allocation::EAS_ALLOCATED;
            tmp.fence = NULL;
            tmp.start = allocs[index].start;
            tmp.end = start;
            allocs[index].start = start;
            allocs.insert(allocs.begin()+index,tmp);
            index++;
        }
    }
    else
    {
        if (mutex)
            mutex->Release();
#ifdef _DEBUG
        os::Printer::log("COMMIT OF UNMAPPED MEMORY ATTEMPTED",ELL_WARNING);
#endif // _DEBUG
        return false;
    }
    for (; index<allocs.size()&&allocs[index].end<=end; index++)
    {
        if (invalidState(allocs[index]))
        {
            if (mutex)
                mutex->Release();
#ifdef _DEBUG
            //os::Printer::log("INVALID STATE IN IGPUTransientBuffer::commit",ELL_ERROR);
#endif // _DEBUG
            return false;
        }

        if (allocs[index].state==Allocation::EAS_ALLOCATED)
        {
            allocs[index].state = Allocation::EAS_PENDING_RENDER_CMD;
        }
        else
        {
            if (mutex)
            {
                lastChanged++;
                allocationChanged->SignalConditionToAll();
                mutex->Release();
            }
#ifdef _DEBUG
            os::Printer::log("COMMIT OF UNMAPPED MEMORY ATTEMPTED",ELL_WARNING);
#endif // _DEBUG
            return false;
        }
    }
    if (index<allocs.size()&&allocs[index].start<end)
    {
        if (allocs[index].state==Allocation::EAS_ALLOCATED)
        {
            if (invalidState(allocs[index]))
            {
                if (mutex)
                {
                    lastChanged++;
                    allocationChanged->SignalConditionToAll();
                    mutex->Release();
                }
#ifdef _DEBUG
                //os::Printer::log("INVALID STATE IN IGPUTransientBuffer::commit",ELL_ERROR);
#endif // _DEBUG
                return false;
            }

            Allocation tmp;
            tmp.state = Allocation::EAS_ALLOCATED;
            tmp.fence = NULL;
            tmp.start = end;
            tmp.end = allocs[index].end;
            allocs[index].end = end;
            allocs[index].state = Allocation::EAS_PENDING_RENDER_CMD;
            index++;
            allocs.insert(allocs.begin()+index,tmp);
        }
        else
        {
            if (mutex)
            {
                lastChanged++;
                allocationChanged->SignalConditionToAll();
                mutex->Release();
            }
#ifdef _DEBUG
            os::Printer::log("COMMIT OF UNMAPPED MEMORY ATTEMPTED",ELL_WARNING);
#endif // _DEBUG
            return false;
        }
    }

    if (mutex)
    {
        lastChanged++;
        allocationChanged->SignalConditionToAll();
        mutex->Release();
    }
    return true;
}
//
bool IGPUTransientBuffer::queryRange(const size_t& start, const size_t& end, const Allocation::E_ALLOCATION_STATE& state)
{
    if (start>end)
        return false;
    else if (start==end)
        return true;

    if (mutex)
        mutex->Get();
    if (!underlyingBuffer||end>underlyingBuffer->getSize()||start>=underlyingBuffer->getSize())
    {
        if (mutex)
            mutex->Release();
        return false;
    }
    uint32_t index;
    if (!findFirstChunk(index,start)||invalidState(allocs[index]))
    {
        if (mutex)
            mutex->Release();
        return false;
    }

    for (; index<allocs.size()&&allocs[index].start<end; index++)
    {
        if (allocs[index].state!=state)
        {
            if (mutex)
                mutex->Release();
            return false;
        }
    }

    if (mutex)
        mutex->Release();
    return true;
}
//
bool IGPUTransientBuffer::Place(size_t &offsetOut, const void* data, const size_t& dataSize, const size_t& alignment, const E_WAIT_POLICY &waitPolicy, const bool &growIfTooSmall)
{
    if (!data||dataSize==0)
    {
        offsetOut = 0xdeadbeefu;
        return true;
    }

    size_t offset;
    if (Alloc(offset,dataSize,alignment,waitPolicy,growIfTooSmall)!=EARS_SUCCESS)
        return false;

    bool result = true;
    if (underlyingBufferAsMapped)
        memcpy(((uint8_t*)underlyingBufferAsMapped->getPointer())+offset,data,dataSize);
    else if (underlyingBuffer->canUpdateSubRange())
        underlyingBuffer->updateSubRange(offset,dataSize,data);
    else
        result = false;

    if (!Commit(offset,offset+dataSize))
    {
        Free(offset,offset+dataSize);
        return false;
    }
    offsetOut = offset;

    return result;
}
//! Unless memory is being used by GPU it will be returned to free pool straight away
//! Useful if you dont end up using comitted memory by GPU
bool IGPUTransientBuffer::Free(const size_t& start, const size_t& end)
{
    if (start>end)
        return false;
    else if (start==end)
        return true;

    if (mutex)
        mutex->Get();
    if (!underlyingBuffer||end>underlyingBuffer->getSize()||start>=underlyingBuffer->getSize())
    {
        if (mutex)
            mutex->Release();
        return false;
    }
    uint32_t index;
    if (!findFirstChunk(index,start)||invalidState(allocs[index]))
    {
        if (mutex)
            mutex->Release();
        //os::Printer::log("INVALID STATE IN IGPUTransientBuffer::Free",ELL_ERROR);
        return false;
    }

    //state of chunks must be EAS_PENDING
    //change state to EAS_FREE
    //pay attention to fences
    if (allocs[index].state==Allocation::EAS_PENDING_RENDER_CMD||allocs[index].state==Allocation::EAS_ALLOCATED)
    {
        if (allocs[index].start<start)
        {
            Allocation tmp;
            tmp.state = allocs[index].state;
            if (allocs[index].fence)
            {
                tmp.fence = allocs[index].fence;
                tmp.fence->grab();
            }
            else
                tmp.fence = NULL;
            tmp.start = allocs[index].start;
            tmp.end = start;
            allocs[index].start = start;
            allocs.insert(allocs.begin()+index,tmp);
            index++;
        }
    }
    else
    {
        if (mutex)
        {
            lastChanged++;
            allocationChanged->SignalConditionToAll();
            mutex->Release();
        }
#ifdef _DEBUG
        os::Printer::log("DOUBLE FREE ATTEMPTED",ELL_WARNING);
#endif // _DEBUG
        return false;
    }
    for (; index<allocs.size()&&allocs[index].end<=end; index++)
    {
        if (allocs[index].state==Allocation::EAS_PENDING_RENDER_CMD||allocs[index].state==Allocation::EAS_ALLOCATED)
        {
            allocs[index].state = Allocation::EAS_FREE;
        }
        else
        {
            if (mutex)
            {
                lastChanged++;
                allocationChanged->SignalConditionToAll();
                mutex->Release();
            }
#ifdef _DEBUG
            os::Printer::log("BAD FREE ATTEMPTED",ELL_WARNING);
#endif // _DEBUG
            return false;
        }
    }
    if (index<allocs.size()&&allocs[index].start<end)
    {
        if (allocs[index].state==Allocation::EAS_PENDING_RENDER_CMD||allocs[index].state==Allocation::EAS_ALLOCATED)
        {
            Allocation tmp;
            tmp.state = allocs[index].state;
            if (allocs[index].fence)
            {
                tmp.fence = allocs[index].fence;
                tmp.fence->grab();
            }
            else
                tmp.fence = NULL;
            tmp.start = end;
            tmp.end = allocs[index].end;
            allocs[index].end = end;
            allocs[index].state = Allocation::EAS_FREE;
            index++;
            allocs.insert(allocs.begin()+index,tmp);
        }
        else
        {
            if (mutex)
            {
                lastChanged++;
                allocationChanged->SignalConditionToAll();
                mutex->Release();
            }
#ifdef _DEBUG
            os::Printer::log("DOUBLE FREE ATTEMPTED",ELL_WARNING);
#endif // _DEBUG
            return false;
        }
    }
    totalFreeSpace += end-start;


    if (mutex)
    {
        lastChanged++;
        allocationChanged->SignalConditionToAll();
        mutex->Release();
    }
    return true;
}
// GPU side calls
bool IGPUTransientBuffer::fenceRangeUsedByGPU(const size_t& start, const size_t& end)
{
    if (start>end)
        return false;
    else if (start==end)
        return true;

    //allocationChanged condition not changed when new fence placed
    //we already had too little sequential memory to allocate,
    //fence will make it even less so pointless to examine again
    if (mutex)
        mutex->Get();
    if (!underlyingBuffer)
    {
        if (mutex)
            mutex->Release();
        return false; // ERROR
    }
    uint32_t index;
    if (!findFirstChunk(index,start)||invalidState(allocs[index]))
    {
        if (mutex)
            mutex->Release();
        //os::Printer::log("INVALID STATE IN IGPUTransientBuffer::fenceRangeUsedByGPU",ELL_ERROR);
        return false;
    }

    IDriverFence* newFence = Driver->placeFence();
    if (!newFence)
    {
        if (mutex)
            mutex->Release();
        //os::Printer::log("CANT PLACE FENCE IN IGPUTransientBuffer::fenceRangeUsedByGPU",ELL_ERROR);
        return false;
    }

    //scan over chunks, chunks must be EAS_PENDING
    //can defragment while going at it
    //pay attention to fences (merging and splitting chunks as necessary)
    //replace old fence with new fence
    if (allocs[index].state!=Allocation::EAS_ALLOCATED)
    {
        if (allocs[index].start<start)
        {
            Allocation tmp;
            tmp.state = allocs[index].state;
            tmp.fence = allocs[index].fence;
            if (tmp.fence)
                tmp.fence->grab();
            tmp.start = allocs[index].start;
            tmp.end = start;
            allocs[index].start = start;
            allocs.insert(allocs.begin()+index,tmp);
            index++;
        }
    }
    else
    {
#ifdef _DEBUG
        os::Printer::log("IGPUTransientBuffer::fenceRangeUsedByGPU ERROR!",ELL_ERROR);
#endif // _DEBUG
        if (mutex)
            mutex->Release();
        newFence->drop();
        return false;
    }
#ifdef _DEBUG
    bool fencedFreedRange = false;
#endif // _DEBUG
    for (; index<allocs.size()&&allocs[index].end<=end; index++)
    {
        if (allocs[index].state==Allocation::EAS_PENDING_RENDER_CMD)
        {
            newFence->grab();
            if (allocs[index].fence)
                allocs[index].fence->drop();
            allocs[index].fence = newFence;
        }
        else if (allocs[index].state==Allocation::EAS_ALLOCATED)
        {
#ifdef _DEBUG
            os::Printer::log("BAD FREE ATTEMPTED",ELL_WARNING);
#endif // _DEBUG
            if (mutex)
                mutex->Release();
            newFence->drop();
            return false;
        }
#ifdef _DEBUG
        else
            fencedFreedRange = true;
#endif // _DEBUG
    }
    if (index<allocs.size()&&allocs[index].start<end)
    {
        if (allocs[index].state==Allocation::EAS_PENDING_RENDER_CMD)
        {
            Allocation tmp;
            tmp.state = allocs[index].state;
            tmp.fence = allocs[index].fence;
            tmp.start = end;
            tmp.end = allocs[index].end;

            newFence->grab();
            allocs[index].fence = newFence;
            allocs[index].end = end;

            index++;
            allocs.insert(allocs.begin()+index,tmp);
        }
        else if (allocs[index].state==Allocation::EAS_ALLOCATED)
        {
#ifdef _DEBUG
            os::Printer::log("BAD FREE ATTEMPTED",ELL_WARNING);
#endif // _DEBUG
            if (mutex)
                mutex->Release();
            newFence->drop();
            return false;
        }
#ifdef _DEBUG
        else
            fencedFreedRange = true;
#endif // _DEBUG
    }

    newFence->drop();
    if (mutex)
        mutex->Release();

#ifdef _DEBUG
    if (fencedFreedRange)
        os::Printer::log("IGPUTransientBuffer::fenceRangeUsedByGPU trying to fence a Free()'d range, not placing a fence. GPU gets incoherent data",ELL_WARNING);
#endif // _DEBUG

    return true;
}
//
bool IGPUTransientBuffer::waitRangeFences(const size_t& start, const size_t& end, size_t timeOutNs)
{
#if defined(_IRR_WINDOWS_API_)
    LARGE_INTEGER nTime;
    QueryPerformanceCounter(&nTime);
    size_t lastMeasuredTime = nTime.QuadPart;
#else
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts); // Works on Linux
    size_t lastMeasuredTime = ts.tv_nsec;
    lastMeasuredTime += ts.tv_sec*1000000000;
#endif

    if (mutex)
        mutex->Get();
    if (!underlyingBuffer)
    {
        if (mutex)
            mutex->Release();
        return true;
    }

#ifdef _EXTREME_DEBUG
    if (!validate_ALREADYMUTEXED())
    {
        os::Printer::log("TRNASIENT BUFFER VALIDATION FAILED!\n",ELL_ERROR);
        PrintDebug(false);
    }
#endif // _EXTREME_DEBUG

    //defragment while checking if all are unmapped
    uint32_t j;
    if (!findFirstChunk(j,start))
    {
        if (mutex)
            mutex->Release();
        return false;
    }
    size_t i=j+1;
    for (; i<allocs.size()&&allocs[i].start<end; i++)
    {
        bool retest = false;
        //not the same as next
        if (allocs[j].state!=allocs[i].state||allocs[j].fence!=allocs[i].fence)
        {
            switch (allocs[j].state)
            {
                case Allocation::EAS_FREE:
                case Allocation::EAS_PENDING_RENDER_CMD:
                    //only wait on fence if free
                    if (allocs[j].fence)
                    {
                        switch (allocs[j].fence->waitCPU(timeOutNs,flushOnWait))
                        {
                            case EDFR_TIMEOUT_EXPIRED:
                                allocs[j].end = allocs[i].start;
                                if (j+1<i)
                                {
                                    j++;
                                    for (; i<allocs.size(); i++,j++)
                                    {
                                        allocs[j] = allocs[i];
                                    }
                                    allocs.resize(j);
                                }
                                if (mutex)
                                    mutex->Release();
                                return false;
                                break;
                            case EDFR_CONDITION_SATISFIED:
                                {
    #if defined(_IRR_WINDOWS_API_)
                                    LARGE_INTEGER nTime;
                                    QueryPerformanceCounter(&nTime);
								    size_t timeMeasuredNs = nTime.QuadPart;
                #else
                                    timespec ts;
                                    clock_gettime(CLOCK_REALTIME, &ts); // Works on Linux
                                    size_t timeMeasuredNs = ts.tv_nsec;
                                    timeMeasuredNs += ts.tv_sec*1000000000;
    #endif
                                    size_t timeDiff = lastMeasuredTime-timeMeasuredNs;
                                    if (timeDiff>timeOutNs)
                                        timeOutNs = 0;
                                    else
                                        timeOutNs -= timeDiff;
                                    lastMeasuredTime = timeMeasuredNs;
                                }
                                break;
                            default: //any other thing
                                allocs[j].fence->drop();
                                allocs[j].fence = NULL;

                                if (allocs[j].state==Allocation::EAS_FREE)
                                    totalTrueFreeSpace += allocs[i].start-allocs[j].start;

                                retest = !allocs[i].fence;
                                break;
                        }
                    }
                    break;
               default:
                    break;
            }

            if (retest)
            {
                i--;
                continue;
            }

            if (j+1<i)
            {
                allocs[j].end = allocs[i].start;
                j++;
                allocs[j] = allocs[i];
            }
            else
                j++;
        } //will be removed (collapsed) so drop reference to fence
        else if (allocs[i].fence)
            allocs[i].fence->drop();
    }

    bool returnTrue = true;
    //process last chunk
    if (allocs[j].state==Allocation::EAS_PENDING_RENDER_CMD||allocs[j].state==Allocation::EAS_FREE)
    {
        if (allocs[j].fence)
        {
            switch (allocs[j].fence->waitCPU(timeOutNs,flushOnWait))
            {
                case EDFR_TIMEOUT_EXPIRED:
                    returnTrue = false;
                    break;
                default: //any other thing
                    allocs[j].fence->drop();
                    allocs[j].fence = NULL;

                    if (allocs[j].state==Allocation::EAS_FREE)
                        totalTrueFreeSpace += allocs[i-1].end-allocs[j].start;

                    break;
            }
        }
    }

    //trim and fix the end
    if (j+1<i)
    {
        allocs[j++].end = allocs[i-1].end;
        for (; i<allocs.size(); i++,j++)
            allocs[j] = allocs[i];
        allocs.resize(j);
    }

#ifdef _EXTREME_DEBUG
    if (!validate_ALREADYMUTEXED())
    {
        os::Printer::log("TRNASIENT BUFFER VALIDATION FAILED!\n",ELL_ERROR);
        PrintDebug(false);
    }
#endif // _EXTREME_DEBUG

    if (mutex)
        mutex->Release();
    return returnTrue;
}
//
void IGPUTransientBuffer::DefragDescriptor()
{
    if (mutex)
        mutex->Get();
    if (!underlyingBuffer)
    {
        if (mutex)
            mutex->Release();
        return;
    }

    if (allocs.size()<2)
    {
        if (mutex)
            mutex->Release();
        return;
    }

    size_t j=0;
    largestFreeChunkSize = 0;
    trueLargestFreeChunkSize = 0;
    size_t freeIntervalSize=0;
    size_t trueFreeIntervalSize=0;
    for (size_t i=1; i<allocs.size(); i++)
    {
        if (allocs[j].state!=allocs[i].state||allocs[j].fence!=allocs[i].fence)
        {
            if (allocs[j].state==Allocation::EAS_FREE)
            {
                freeIntervalSize += allocs[i].start-allocs[j].start;
                if (allocs[j].fence)
                {
                    trueFreeIntervalSize += allocs[i].start-allocs[j].start;
                }
                else
                {
                    if (trueFreeIntervalSize>trueLargestFreeChunkSize)
                        trueLargestFreeChunkSize = trueFreeIntervalSize;
                    trueFreeIntervalSize = 0;
                }
            }
            else
            {
                if (trueFreeIntervalSize>trueLargestFreeChunkSize)
                    trueLargestFreeChunkSize = trueFreeIntervalSize;
                if (freeIntervalSize>largestFreeChunkSize)
                    largestFreeChunkSize = freeIntervalSize;
                freeIntervalSize = 0;
                trueFreeIntervalSize = 0;
            }

            if (j+1<i)
            {
                allocs[j].end = allocs[i].start;
                j++;
                allocs[j] = allocs[i];
            }
            else
                j++;
            continue;
        }

        if (allocs[i].fence)
            allocs[i].fence->drop();
    }

    if (allocs[j].state==Allocation::EAS_FREE)
    {
        freeIntervalSize += allocs.back().end-allocs[j].start;
        if (allocs[j].fence)
            trueFreeIntervalSize += allocs.back().end-allocs[j].start;
    }
    if (trueFreeIntervalSize>trueLargestFreeChunkSize)
        trueLargestFreeChunkSize = trueFreeIntervalSize;
    if (freeIntervalSize>largestFreeChunkSize)
        largestFreeChunkSize = freeIntervalSize;

    if (j+1<allocs.size())
    {
        allocs[j].end = allocs.back().end;
        allocs.resize(j+1);
    }

    if (mutex)
        mutex->Release();
}
