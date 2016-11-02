#ifndef __I_META_GRANULAR_BUFFER_H__
#define __I_META_GRANULAR_BUFFER_H__
#include "assert.h"
#include <algorithm>

namespace irr
{
namespace video
{
class IGPUMappedBuffer;
}
}

#include "ICPUBuffer.h"
#include "IVideoDriver.h"

// 20% slower
//#define _IRR_SWAP_BY_RECREATE_

namespace irr
{
namespace core
{

template <class T>
class IMetaGranularBuffer : public virtual IReferenceCounted
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

        inline void ValidateHashMap(const uint32_t& maxAlloc)
        {
            for (size_t i=0; i<Granules; i++)
            {
                uint32_t key = residencyRedirectTo[i];
                if (key==0xdeadbeefu)
                    continue;

                assert(key<maxAlloc);
                for (size_t j=0; j<i; j++)
                {
                    assert(key!=residencyRedirectTo[j]);
                }
            }
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
            if (!B)
            {
                Granules = 0;
                if (residencyRedirectTo)
                    free(residencyRedirectTo);
                residencyRedirectTo = NULL;
            }
        }
        virtual ~IMetaGranularBuffer()
        {
            if (B)
                B->drop();

            if (residencyRedirectTo)
                free(residencyRedirectTo);
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
                if (!GrowBackBuffer(newAllocCount))
                {
                    if (residencyRedirectTo)
                    {
                        free(residencyRedirectTo);
                        residencyRedirectTo = NULL;
                    }
                    Allocated = 0;
                    Granules = 0;
                    for (size_t i=0; i<count; i++)
                        granuleIDs[i] = 0xdeadbeefu;
                    return false;
                }
            }
            //allocate more IDs if needed
            if (newAllocCount>Granules)
            {
                size_t pseudoNewAllocCount = Allocated+count;
                newAllocCount = pseudoNewAllocCount+BackBufferGrowStep-1;
                residencyRedirectTo = (uint32_t*)realloc(residencyRedirectTo,newAllocCount*4);

                if (!residencyRedirectTo)
                {
                    Allocated = 0;
                    Granules = 0;
                    for (size_t i=0; i<count; i++)
                        granuleIDs[i] = 0xdeadbeefu;
                    return false;
                }

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
                core::heapsort<uint32_t>(indicesTmp,count);

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

        inline const size_t getGranuleByteSize() const {return GranuleByteSize;}
};


class IMetaGranularCPUBuffer : public IMetaGranularBuffer<core::ICPUBuffer>
{
    protected:
        core::ICPUBuffer* A;
    public:
        IMetaGranularCPUBuffer(const size_t& granuleSize, const size_t& granuleCount, const bool& clientMemeory=true, const size_t& bufferGrowStep=512, const size_t& bufferShrinkStep=2048)
                                    : IMetaGranularBuffer<core::ICPUBuffer>(granuleSize,granuleCount,bufferGrowStep,bufferShrinkStep), A(NULL)
        {
            if (!B)
                return;

            A = new core::ICPUBuffer(GranuleByteSize*granuleCount);
            if (!A)
            {
                B->drop();
                B = NULL;

                Granules = 0;
                if (residencyRedirectTo)
                    free(residencyRedirectTo);
                residencyRedirectTo = NULL;
            }
        }

        virtual ~IMetaGranularCPUBuffer()
        {
            if (A)
                A->drop();
        }

        virtual core::ICPUBuffer* getFrontBuffer() {return A;}

        virtual void SwapBuffers(void (*StuffToDoToNewBuffer)(core::ICPUBuffer*,void*)=NULL, void* userData=NULL)
        {
            if (A->getSize()!=B->getSize())
            {
                core::ICPUBuffer* C = new ICPUBuffer(B->getSize());
                if (!C)
                    return;

                A->drop();
                A = C;
            }
            if (Allocated)
                memcpy(A->getPointer(),B->getPointer(),Allocated*GranuleByteSize);

            if (StuffToDoToNewBuffer)
                StuffToDoToNewBuffer(A,userData);
        }
};
}


namespace video
{

class IMetaGranularGPUMappedBuffer : public core::IMetaGranularBuffer<video::IGPUBuffer>
{
        video::IVideoDriver* Driver;
        video::IGPUBuffer* A;

        const bool InClientMemeory;
    public:
        IMetaGranularGPUMappedBuffer(video::IVideoDriver* driver, const size_t& granuleSize, const size_t& granuleCount, const bool& clientMemeory=true, const size_t& bufferGrowStep=512, const size_t& bufferShrinkStep=2048)
                                    : core::IMetaGranularBuffer<video::IGPUBuffer>(granuleSize,granuleCount,bufferGrowStep,bufferShrinkStep), Driver(driver), A(NULL), InClientMemeory(clientMemeory)
        {
            if (!B)
                return;

            A = Driver->createGPUBuffer(GranuleByteSize*granuleCount,NULL,true,InClientMemeory);
            if (!A)
            {
                B->drop();
                B = NULL;

                Granules = 0;
                if (residencyRedirectTo)
                    free(residencyRedirectTo);
                residencyRedirectTo = NULL;
            }
        }
        virtual ~IMetaGranularGPUMappedBuffer()
        {
            if (A)
                A->drop();
        }

        virtual video::IGPUBuffer* getFrontBuffer() {return A;}

        virtual void SwapBuffers(void (*StuffToDoToNewBuffer)(video::IGPUBuffer*,void*)=NULL,void* userData=NULL)
        {
#ifdef _IRR_SWAP_BY_RECREATE_
            A->clandestineRecreate(Allocated*GranuleByteSize,B->getPointer());
#else
            if (A->getSize()!=B->getSize())
                A->reallocate(B->getSize(),false,true);

            if (Allocated)
                A->updateSubRange(0,Allocated*GranuleByteSize,B->getPointer());
#endif // _IRR_SWAP_BY_RECREATE_
            if (StuffToDoToNewBuffer)
                StuffToDoToNewBuffer(A,userData);
        }
};

}
}

#endif
