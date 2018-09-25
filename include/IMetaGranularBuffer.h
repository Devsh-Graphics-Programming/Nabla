#ifndef __I_META_GRANULAR_BUFFER_H__
#define __I_META_GRANULAR_BUFFER_H__

#include "irr/core/alloc/ContiguousPoolAddressAllocator.h"
#include "irr/video/CDoubleBufferingAllocator.h"

#include "irr/core/ICPUBuffer.h"
#include "irr/video/CDoubleBufferingAllocator.h"

namespace irr
{
namespace core
{

template <class T>
class IMetaGranularBuffer : public virtual core::IReferenceCounted
{
    protected:
        size_t Allocated;
        size_t Granules;

        uint32_t* residencyRedirectTo;

        const size_t GranuleByteSize;
        const size_t BackBufferGrowStep;
        const size_t BackBufferShrinkStep;

        core::ICPUBuffer* B;

        inline bool GrowBackBuffer(const size_t& newGranuleCount)
        {
            core::ICPUBuffer* C = new core::ICPUBuffer(newGranuleCount*GranuleByteSize);
            if (!C)
                return false;

            if (Allocated)
                memcpy(C->getPointer(),B->getPointer(),Allocated*GranuleByteSize);
            B->drop();
            B = C;

            return true;
        }
        inline void ShrinkBackBuffer()
        {
            size_t newGranules = Allocated+BackBufferGrowStep;
            core::ICPUBuffer* C = new core::ICPUBuffer(newGranules*GranuleByteSize);
            if (!C)
                return;

            if (Allocated)
                memcpy(C->getPointer(),B->getPointer(),Allocated*GranuleByteSize);
            B->drop();
            B = C;
        }
        virtual ~IMetaGranularBuffer()
        {
            if (B)
                B->drop();

            if (residencyRedirectTo)
                free(residencyRedirectTo);
        }
    public:
        IMetaGranularBuffer(const size_t& granuleSize, const size_t& granuleCount, const size_t& bufferGrowStep=512, const size_t& bufferShrinkStep=2048)
                            :   Allocated(0), Granules(granuleCount), GranuleByteSize(granuleSize), residencyRedirectTo(NULL),
                                BackBufferGrowStep(bufferGrowStep), BackBufferShrinkStep(bufferShrinkStep), B(NULL)
        {
            residencyRedirectTo = (uint32_t*)malloc(Granules*4);
            if (!residencyRedirectTo)
                return;

            for (size_t i=0; i<Granules; i++)
                residencyRedirectTo[i] = 0xdeadbeefu;

            B = new core::ICPUBuffer(GranuleByteSize*granuleCount);
        }

        virtual T* getFrontBuffer() = 0;

        inline void* getBackBufferPointer() {return B->getPointer();}

        //Makes Writes visible
        virtual void SwapBuffers(void (*StuffToDoToNewBuffer)(T*,void*)=NULL,void* userData=NULL) = 0;

        inline const size_t& getAllocatedCount() const {return Allocated;}

        inline const size_t& getCapacity() {return Granules;}

        /// Preconditions:
        ///     1) no holes in allocation
        ///     2) can have holes in redirects
        virtual bool Alloc(uint32_t* granuleIDs, const size_t& count)
        {
//			ValidateHashMap(Allocated);
            size_t allocationsCount = 0;

            size_t newAllocCount = Allocated+count;
            if (newAllocCount*GranuleByteSize>B->getSize())
            {
                newAllocCount += BackBufferGrowStep-1;
                //grow data store
                GrowBackBuffer(newAllocCount);
            }
            //allocate more IDs if needed
            if (newAllocCount>Granules)
            {
                size_t pseudoNewAllocCount = Allocated+count;
                newAllocCount = pseudoNewAllocCount+BackBufferGrowStep-1;
                residencyRedirectTo = (uint32_t*)realloc(residencyRedirectTo,newAllocCount*4);

                while (allocationsCount<count&&Granules<pseudoNewAllocCount)
                {
                    residencyRedirectTo[Granules] = Allocated++;
                    granuleIDs[allocationsCount++] = Granules;
                    Granules++;
                }
                //ValidateHashMap(Allocated);
                while (Granules<newAllocCount)
                {
                    residencyRedirectTo[Granules++] = 0xdeadbeefu;
                }
            }
            else
            {
                size_t diff = Granules-newAllocCount;
                if (diff>BackBufferShrinkStep)
                {
                    size_t tmp = Granules-1;
                    for (size_t i=0; i<diff; i++,tmp--)
                    {
                        if (residencyRedirectTo[tmp]>=0xdeadbeefu)
                            break;
                    }

                    if (Granules-tmp>BackBufferGrowStep)
                    {
                        Granules = tmp+1;
                        residencyRedirectTo = (uint32_t*)realloc(residencyRedirectTo,Granules*4);
                    }
                }
            }
//            ValidateHashMap(Allocated);

            for (size_t i=0; allocationsCount<count; i++)
            {
                if (residencyRedirectTo[i]>=0xdeadbeefu)
                {
                    residencyRedirectTo[i] = Allocated;
                    Allocated++;
                    granuleIDs[allocationsCount++] = i;
                }
            }
//            ValidateHashMap(Allocated);

            return true;
        }

        inline const uint32_t& getRedirectFromID(const size_t& ix) const
        {
#ifdef _DEBUG
            assert(ix<Granules);
#endif // _DEBUG
            return residencyRedirectTo[ix];
        }


        virtual void Free(const uint32_t* granuleIDs, const size_t& count)
        {
            if (count==0)
                return;

            uint32_t* indicesTmp = new uint32_t[count];
            for (size_t i=0; i<count; i++)
            {
                indicesTmp[i] = residencyRedirectTo[granuleIDs[i]];
#ifdef _DEBUG
                assert(indicesTmp[i]<0xdeadbeefu);
#endif // _DEBUG
            }
            if (count>1)
            {
                std::make_heap(indicesTmp,indicesTmp+count);
                std::sort_heap(indicesTmp,indicesTmp+count);
            }

            size_t deletedGranuleCnt=0;
//            ValidateHashMap(Allocated);
            for (size_t i=0; i<Granules; i++)
            {
                uint32_t rfrnc = residencyRedirectTo[i];
                if (rfrnc>=0xdeadbeefu)
                    continue;

                uint32_t* ptr = std::lower_bound(indicesTmp,indicesTmp+count,rfrnc);
                if (ptr<(indicesTmp+count)&&ptr[0]==rfrnc)
                {
                    deletedGranuleCnt++;
                    residencyRedirectTo[i] = 0xdeadbeefu;
                }
                else
                {
                    size_t difference = ptr-indicesTmp;
                    residencyRedirectTo[i] = rfrnc-difference;
                }
            }
#ifdef _DEBUG
			assert(deletedGranuleCnt==count);
#endif // _DEBUG
//            ValidateHashMap(Allocated-count);

            if (deletedGranuleCnt)
            {
                uint8_t* basePtr = reinterpret_cast<uint8_t*>(this->getBackBufferPointer());
                size_t nextIx=1;
                size_t j=0;
                while (nextIx<deletedGranuleCnt)
                {
                    size_t len = indicesTmp[nextIx]-indicesTmp[j]-1;
                    if (len)
                    {
                        uint8_t* tmpPtr = basePtr+indicesTmp[j]*GranuleByteSize;
                        memmove(tmpPtr-j*GranuleByteSize,tmpPtr+GranuleByteSize,len*GranuleByteSize);
                    }
                    j = nextIx++;
                }
                size_t len = Allocated-indicesTmp[j]-1;
                if (len)
                {
                    uint8_t* tmpPtr = basePtr+indicesTmp[j]*GranuleByteSize;
                    memmove(tmpPtr-j*GranuleByteSize,tmpPtr+GranuleByteSize,len*GranuleByteSize);
                }
            }

            delete [] indicesTmp;
            Allocated -= deletedGranuleCnt;

            if (B->getSize()/GranuleByteSize-Allocated>=BackBufferShrinkStep+BackBufferGrowStep)
                ShrinkBackBuffer();

//			ValidateHashMap(Allocated);
        }
};

}


namespace video
{

class IMetaGranularGPUMappedBuffer : public core::IMetaGranularBuffer<video::IGPUBuffer>
{
    protected:
        video::CDoubleBufferingAllocator<core::ContiguousPoolAddressAllocatorST<uint32_t>,true>* alloc;

        virtual ~IMetaGranularGPUMappedBuffer()
        {
            if (alloc)
                alloc->drop();
            if (A)
                A->drop();
        }

        video::IVideoDriver* Driver;
        video::IGPUBuffer* A;

        const bool InClientMemeory;

        static video::IGPUBuffer* createUpBuffer(video::IVideoDriver* driver, const size_t& granuleSize, const size_t& granuleCount)
        {
            auto tmp = driver->createUpStreamingGPUBufferOnDedMem(granuleSize*granuleCount);
            tmp->getBoundMemory()->mapMemoryRange(video::IDriverMemoryAllocation::EMCAF_WRITE,video::IDriverMemoryAllocation::MemoryRange(0u,granuleSize*granuleCount));
            return tmp;
        }
        static video::IGPUBuffer* createGPUBuffer(video::IVideoDriver* driver, const size_t& granuleSize, const size_t& granuleCount)
        {
            ///return driver->createDeviceLocalGPUBufferOnDedMem(granuleSize*granuleCount);
	        IDriverMemoryBacked::SDriverMemoryRequirements reqs;
	        reqs.vulkanReqs.size = granuleSize*granuleCount;
	        reqs.vulkanReqs.alignment = 0;
	        reqs.vulkanReqs.memoryTypeBits = 0xffffffffu;
	        reqs.memoryHeapLocation = IDriverMemoryAllocation::ESMT_DEVICE_LOCAL;
	        reqs.mappingCapability = IDriverMemoryAllocation::EMCF_CANNOT_MAP;
	        reqs.prefersDedicatedAllocation = true;
	        reqs.requiresDedicatedAllocation = true;
            return driver->createGPUBufferOnDedMem(reqs,true);
        }
    public:
        IMetaGranularGPUMappedBuffer(video::IVideoDriver* driver, const size_t& granuleSize, const size_t& granuleCount, const bool& clientMemeory=true, const size_t& bufferGrowStep=512, const size_t& bufferShrinkStep=2048)
                                    : core::IMetaGranularBuffer<video::IGPUBuffer>(granuleSize,granuleCount,bufferGrowStep,bufferShrinkStep),
                                    Driver(driver), A(createGPUBuffer(driver,granuleSize,granuleCount)), InClientMemeory(clientMemeory)
        {
            alloc = new video::CDoubleBufferingAllocator<core::ContiguousPoolAddressAllocatorST<uint32_t>,true>(driver,video::IDriverMemoryAllocation::MemoryRange(0u,granuleSize*granuleCount),
                                                                                                                createUpBuffer(driver,granuleSize,granuleCount),0u,A,granuleSize);
        }

        virtual video::IGPUBuffer* getFrontBuffer() {return A;}

        virtual void SwapBuffers(void (*StuffToDoToNewBuffer)(video::IGPUBuffer*,void*)=NULL,void* userData=NULL)
        {
            if (A->getSize()!=B->getSize())
            {
                IDriverMemoryBacked::SDriverMemoryRequirements reqs = A->getMemoryReqs();
                reqs.vulkanReqs.size = B->getSize();
                {auto rep = Driver->createGPUBufferOnDedMem(reqs,true); A->pseudoMoveAssign(rep); rep->drop();}
            }

            if (Allocated)
                A->updateSubRange(IDriverMemoryAllocation::MemoryRange(0,Allocated*GranuleByteSize),B->getPointer());

            if (StuffToDoToNewBuffer)
                StuffToDoToNewBuffer(A,userData);
        }
};

}
}

#endif
