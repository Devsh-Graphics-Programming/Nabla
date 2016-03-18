#include "IGPUTransientBuffer.h"
#include "COpenGLPersistentlyMappedBuffer.h"
#include "os.h"
#include "FW_Mutex.h"

using namespace irr;
using namespace video;



IGPUTransientBuffer::IGPUTransientBuffer(IVideoDriver* driver, const size_t& bufsize, const bool& inCPUMem, const bool& growable, const bool& threadSafe) : lastChanged(0), canGrow(growable), Driver(driver)
{
    Allocation first;
    first.state = Allocation::EAS_FREE;
    first.fence = NULL;
    first.start = 0;
    first.end = bufsize;
    allocs.reserve(1024);
    allocs.push_back(first);

    underlyingBuffer = Driver->createPersistentlyMappedBuffer(bufsize,NULL,EGBA_WRITE,true,inCPUMem);

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

IGPUTransientBuffer::E_ALLOC_RETURN_STATUS IGPUTransientBuffer::Alloc(size_t &offsetOut, const size_t &maxSize, E_WAIT_POLICY waitPolicy, bool growIfTooSmall)
{
    if (maxSize==0)
        return EARS_FAIL;


    if (!canGrow)
        growIfTooSmall = false;/*
    else if (growIfTooSmall&&waitPolicy==EWP_DONT_WAIT)
        waitPolicy = EWP_WAIT_FOR_CPU_UNMAP;*/

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

#ifdef _DEBUG
    if (!validate_ALREADYMUTEXED())
        os::Printer::log("TRNASIENT BUFFER VALIDATION FAILED!\n",ELL_ERROR);
#endif // _DEBUG

    // defragment all the time
    // grow if everything is unmapped
    // circle releasing fence
    do
    {
        bool allUnmapped = true;
        bool noFencesToCycle = true;
        bool allFree = true;
        //defragment while checking if all are unmapped
        size_t j=0;
        for (size_t i=1; i<allocs.size(); i++)
        {
            bool retest = false;
            //not the same as next
            if (allocs[j].state!=allocs[i].state&&allocs[j].fence!=allocs[i].fence)
            {
                switch (allocs[j].state)
                {
                    case Allocation::EAS_FREE:
                        if (allocs[j].fence)
                        {
                            switch (allocs[j].fence->waitCPU(0))
                            {
                                case EDFR_TIMEOUT_EXPIRED:
                                    noFencesToCycle = false;
                                    break;
                                default: //any other thing
                                    allocs[j].fence->drop();
                                    allocs[j].fence = NULL;
                                    retest = !allocs[i].fence;
                                    break;
                            }
                        }
                        //could have changed
                        if (!allocs[j].fence)
                        {
                            if (allocs[i].start-allocs[j].start>=maxSize)
                            {
                                allocs[j].state = Allocation::EAS_ALLOCATED;
                                offsetOut = allocs[j].start;
                                if (j<i-1)
                                {
                                    allocs[j].end = allocs[i].start;
                                    j++;i++;
                                    for (; i<allocs.size(); i++,j++)
                                    {
                                        allocs[j] = allocs[i];
                                    }
                                    allocs.resize(j);
                                }
                                if (mutex)
                                    mutex->Release();
                                if (waitPolicy&&allocMutex)
                                    allocMutex->Release();
                                return EARS_SUCCESS;
                            }
                        }
                        break;
                    case Allocation::EAS_ALLOCATED:
                        allUnmapped = false;
                        ///allFree = false; //implicit in later check
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

                if (j<i-1)
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

        if (j+1<allocs.size())
        {
            allocs[j].end = allocs.back().end;
            allocs.resize(j+1);
        }

        if (allocs[j].state==Allocation::EAS_FREE)
        {
            if (allocs[j].fence)
            {
                switch (allocs[j].fence->waitCPU(0))
                {
                    case EDFR_TIMEOUT_EXPIRED:
                        noFencesToCycle = false;
                        break;
                    default: //any other thing
                        allocs[j].fence->drop();
                        allocs[j].fence = NULL;
                        break;
                }
            }
            //could have changed
            if (!allocs[j].fence&&allocs[j].end-allocs[j].start>=maxSize)
            {
                allocs[j].state = Allocation::EAS_ALLOCATED;
                offsetOut = allocs[j].start;
                if (mutex)
                    mutex->Release();
                if (waitPolicy&&allocMutex)
                    allocMutex->Release();
                return EARS_SUCCESS;
            }
        }
        else if (allocs[j].state==Allocation::EAS_ALLOCATED)
        {
            allUnmapped = false;
            ///allFree = false; //implicit in later check
        }
        else
            allFree = false;

#ifdef _DEBUG
        if (!validate_ALREADYMUTEXED())
            os::Printer::log("TRNASIENT BUFFER VALIDATION FAILED!\n",ELL_ERROR);
#endif // _DEBUG

        if (allUnmapped)
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
            else if (allFree&&(waitPolicy<EWP_WAIT_FOR_GPU_FREE||noFencesToCycle))
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
bool IGPUTransientBuffer::Place(size_t &offsetOut, void* data, const size_t& dataSize, const E_WAIT_POLICY &waitPolicy, const bool &growIfTooSmall)
{
    if (!data||dataSize==0)
    {
        offsetOut = 0xdeadbeefu;
        return true;
    }

    size_t offset;
    if (!Alloc(offset,dataSize,waitPolicy,growIfTooSmall))
        return false;

    memcpy(((uint8_t*)underlyingBuffer->getPointer())+offset,data,dataSize);
    if (!Commit(offset,offset+dataSize))
    {
        Free(offset,offset+dataSize);
        return false;
    }
    offsetOut = offset;

    return true;
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
    if (allocs[index].state==Allocation::EAS_PENDING_RENDER_CMD)
    {
        if (allocs[index].start<start)
        {
            Allocation tmp;
            tmp.state = Allocation::EAS_PENDING_RENDER_CMD;
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
            mutex->Release();
#ifdef _DEBUG
        os::Printer::log("DOUBLE FREE ATTEMPTED",ELL_WARNING);
#endif // _DEBUG
        return false;
    }
    for (; index<allocs.size()&&allocs[index].end<=end; index++)
    {
        if (allocs[index].state==Allocation::EAS_PENDING_RENDER_CMD)
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
        if (allocs[index].state==Allocation::EAS_PENDING_RENDER_CMD)
        {
            Allocation tmp;
            tmp.state = Allocation::EAS_PENDING_RENDER_CMD;
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
    for (size_t i=1; i<allocs.size(); i++)
    {
        if (allocs[j].state!=allocs[i].state&&allocs[j].fence!=allocs[i].fence)
        {
            if (j<i-1)
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

    if (j+1<allocs.size())
    {
        allocs[j].end = allocs.back().end;
        allocs.resize(j+1);
    }

    if (mutex)
        mutex->Release();
}
