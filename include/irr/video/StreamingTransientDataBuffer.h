#ifndef __IRR_STREAMING_TRANSIENT_DATA_BUFFER_H__
#define __IRR_STREAMING_TRANSIENT_DATA_BUFFER_H__


#include "irr/core/IReferenceCounted.h"
#include "irr/core/alloc/GeneralpurposeAddressAllocator.h"
#include "irr/core/alloc/HeterogenousMemoryAddressAllocatorAdaptor.h"
#include "irr/video/GPUMemoryAllocatorBase.h"

namespace irr
{
namespace video
{


//! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
class StreamingGPUBufferAllocator : public GPUMemoryAllocatorBase
{
        IDriverMemoryBacked::SDriverMemoryRequirements  mBufferMemReqs;
        std::pair<uint8_t*,IGPUBuffer*>                 lastAllocation;

        inline auto     createAndMapBuffer()
        {
            decltype(lastAllocation) retval; retval.first = nullptr;
            retval.second = mDriver->createGPUBufferOnDedMem(mBufferMemReqs,false);

            auto rangeToMap = IDriverMemoryAllocation::MemoryRange{0u,mBufferMemReqs.vulkanReqs.size};
            auto memory = const_cast<IDriverMemoryAllocation*>(retval.second->getBoundMemory());
            auto mappingCaps = mBufferMemReqs.mappingCapability&IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;
            retval.first  = reinterpret_cast<uint8_t*>(memory->mapMemoryRange(static_cast<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG>(mappingCaps),rangeToMap));

            return retval;
        }
    public:
        StreamingGPUBufferAllocator(IVideoDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs) :
                        GPUMemoryAllocatorBase(inDriver), mBufferMemReqs(bufferReqs), lastAllocation(nullptr,nullptr)
        {
            assert(mBufferMemReqs.mappingCapability&IDriverMemoryAllocation::EMCAF_READ_AND_WRITE); // have to have mapping access to the buffer!
        }

        inline void*        allocate(size_t bytes) noexcept
        {
        #ifdef _DEBUG
            assert(!lastAllocation.first && !lastAllocation.second);
        #endif // _DEBUG
            mBufferMemReqs.vulkanReqs.size = bytes;
            lastAllocation = createAndMapBuffer();
            return lastAllocation.first;
        }

        template<class AddressAllocator>
        inline void*        reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets, bool dontCopyBuffers=false) noexcept
        {
        #ifdef _DEBUG
            assert(lastAllocation.first && lastAllocation.second);
        #endif // _DEBUG

            // set up new size
            auto oldSize = mBufferMemReqs.vulkanReqs.size;
            mBufferMemReqs.vulkanReqs.size = bytes;
            //allocate new buffer
            auto tmp = createAndMapBuffer();

            //move contents
            if (!dontCopyBuffers)
            {
                // only first buffer is bound to allocator
                size_t oldOffset = allocToQueryOffsets.get_align_offset();
                size_t newOffset = AddressAllocator::aligned_start_offset(reinterpret_cast<size_t>(tmp.first),allocToQueryOffsets.max_alignment());
                auto copyRangeLen = std::min(oldSize-oldOffset,bytes-newOffset);

                if ((lastAllocation.second->getBoundMemory()->getCurrentMappingCaps()&IDriverMemoryAllocation::EMCAF_READ) &&
                    (tmp.second->getBoundMemory()->getCurrentMappingCaps()&IDriverMemoryAllocation::EMCAF_WRITE)) // can read from old and write to new
                {
                    memcpy(tmp.first+newOffset,lastAllocation.first+oldOffset,copyRangeLen);
                }
                else
                    mDriver->copyBuffer(lastAllocation.second,tmp.second,oldOffset,newOffset,copyRangeLen);
            }

            //swap the internals of buffers
            const_cast<IDriverMemoryAllocation*>(lastAllocation.second->getBoundMemory())->unmapMemory();
            lastAllocation.second->pseudoMoveAssign(tmp.second);
            tmp.second->drop();

            //book-keeping and return
            lastAllocation.first = tmp.first;
            return lastAllocation.first;
        }

        inline void         deallocate(void* addr) noexcept
        {
        #ifdef _DEBUG
            assert(lastAllocation.first && lastAllocation.second);
        #endif // _DEBUG
            lastAllocation.first = nullptr;
            const_cast<IDriverMemoryAllocation*>(lastAllocation.second->getBoundMemory())->unmapMemory();
            lastAllocation.second->drop();
            lastAllocation.second = nullptr;
        }


        // extras
        inline IGPUBuffer*  getAllocatedBuffer()
        {
            return lastAllocation.second;
        }

        inline void*        getAllocatedPointer()
        {
            return lastAllocation.first;
        }
};


template< typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t> >
class StreamingTransientDataBufferST : protected core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor, public virtual core::IReferenceCounted
{
    protected:
        typedef core::GeneralpurposeAddressAllocator<_size_type>                                                        BasicAddressAllocator;
        core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,StreamingGPUBufferAllocator,CPUAllocator> mAllocator; // no point for a streaming buffer to grow
    public:
        typedef typename BasicAddressAllocator::size_type           size_type;
        static constexpr size_type                                  invalid_address = BasicAddressAllocator::invalid_address;

        //!
        /**
        \param default minAllocSize has been carefully picked to reflect the lowest nonCoherentAtomSize under Vulkan 1.1 which is not 1u .*/
        StreamingTransientDataBufferST(IVideoDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs,
                                       const CPUAllocator& reservedMemAllocator, size_type bufSz, size_type minAllocSize=64u) :
                                                mAllocator(reservedMemAllocator,StreamingGPUBufferAllocator(inDriver,bufferReqs),bufSz,minAllocSize)
        {
        }

        template<typename... Args>
        StreamingTransientDataBufferST(IVideoDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs,
                                       const CPUAllocator& reservedMemAllocator, size_type bufSz, Args&&... args) :
                                                mAllocator(reservedMemAllocator,StreamingGPUBufferAllocator(inDriver,bufferReqs),bufSz,std::forward<Args>(args)...)
        {
        }

        virtual ~StreamingTransientDataBufferST() {}


        inline bool         needsManualFlushOrInvalidate() const {return !(getBuffer()->getMemoryReqs().mappingCapability&video::IDriverMemoryAllocation::EMCF_COHERENT);}

        inline IGPUBuffer*  getBuffer() noexcept {return core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor::getDataAllocator(mAllocator).getAllocatedBuffer();}

        inline void*        getBufferPointer() noexcept {return core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor::getDataAllocator(mAllocator).getAllocatedPointer();}

        // have to try each fence once (need a function for that)
        // but only try a few fences before the next alloc attempt
        // hpw many fences to try at each step?
        template<typename... Args>
        inline size_type    multi_alloc(uint32_t count, size_type* outAddresses, const size_type* bytes, const size_type* alignment, const Args&... args) noexcept
        {
            bool lastTimeAllFree = false;
            while (true)
            {
                mAllocator.multi_alloc_addr();

                size_type unallocatedSize = 0;
                for ()
                {
                    if (outAddr!=invalid_address)
                        continue;

                    unallocatedSize += ;
                }

                if (!unallocatedSize)
                    return true;

                if (execute_frees(8000u)) // 8us
                {
                    if (lastTimeAllFree)
                        break;

                    lastTimeAllFree = true;
                }
                else
                    lastTimeAllFree = false;
            }
        }

        template<typename... Args>
        inline size_type    multi_place(uint32_t count, const void* const* dataToPlace, size_type* outAddresses, const size_type* bytes, const size_type* alignment, const Args&... args) noexcept
        {
        #ifdef _DEBUG
            assert(GPUBuffer has write mapping flags);
        #endif // _DEBUG
            multi_alloc(count,outAddresses,bytes,alignment,args...);
            // fill with data
            for (uint32_t i=0; i<count; i++)
            {
                if (outAddresses[i]!=invalid_address)
                    memcpy(reinterpret_cast<uint8_t*>(getBufferPointer())+outAddresses[i],dataToPlace[i],bytes[i]);
            }
        }

        inline void         multi_free(uint32_t count, const size_type* addr, const size_type* bytes, IDriverFence* fence) noexcept
        {
            if (fence)
                deferredFrees.emplace_back(GPUEventWrapper(fence),DeferredFreeFunctor(mAllocator,count,addr,bytes));
            else
                mAllocator.multi_free_addr(count,addr,bytes);
        }

        //! Returns if all fences have completed
        inline bool         execute_frees(uint64_t timeOutNs=0u)
        {
            auto timeOutPoint = std::chrono::high_resolution_clock::now()+std::chrono::nanoseconds(timeOutNs);

            std::sort(deferredFrees.begin(),deferredFrees.end(),deferredFreeCompareFunc);

            lastMeasuredTime
            // iterate through unique fences
            for (auto it = deferredFrees.begin(); it!=deferredFrees.end();)
            {
                // maybe wait
                switch (allocs[j].fence->waitCPU(timeOutNs,false))
                {
                    case EDFR_TIMEOUT_EXPIRED:
                        return false;
                        break;
                    case EDFR_CONDITION_SATISFIED:
                    default: //any other thing
                        cannotFree = false;
                        break;
                }

                // decrease our wait period
                size_t timeDiff = lastMeasuredTime-timeMeasuredNs; //! TODO: c++11 nanosecond epoch elapse
                if (timeDiff>timeOutNs)
                    timeOutNs = 0;
                else
                    timeOutNs -= timeDiff;
                lastMeasuredTime = timeMeasuredNs;

                auto next = std::upper_bound(it,deferredFrees.end(),deferredFreeCompareFunc);
                // free all complete
                for (auto it2=it; it2!=next; it2++)
                {
                    it2->first->drop();
                    ///totalTrueFreeSpace += it->second.length;
                    mAllocator.multi_free_addr(it->second.offset,it->second.length);
                }
                it = deferredFrees.erase(it,next);
            }
            return true;
        }
    protected:
        class DeferredFreeFunctor : protected core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor
        {
            public:
                DeferredFreeFunctor(core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,StreamingGPUBufferAllocator,CPUAllocator>* alloctr,
                                    size_type numAllocsToFree, const size_type* addrs, const size_type* bytes) : allocRef(alloctr), numAllocs(numAllocsToFree)
                {
                    rangeData = reinterpret_cast<size_type*>(getHostAllocator(*allocRef).allocate(sizeof(size_type)*numAllocs*2u,sizeof(size_type)));
                    memcpy(rangeData            ,addrs,sizeof(size_type)*numAllocs);
                    memcpy(rangeData+numAllocs  ,bytes,sizeof(size_type)*numAllocs);
                }
                DeferredFreeFunctor(DeferredFreeFunctor&& other)
                {
                    allocRef    = other.allocRef;
                    numAllocs   = other.numAllocs;
                    rangeData   = other.rangeData;
                    other.allocRef  = nullptr;
                    other.numAllocs = 0u;
                    other.rangeData = nullptr;
                }
                DeferredFreeFunctor(const DeferredFreeFunctor& other) = delete;

                ~DeferredFreeFunctor()
                {
                    if (rangeData)
                        getHostAllocator(*allocRef).deallocate(rangeData,sizeof(size_type)*numAllocs*2u);
                }

                inline void operator()
                {
                    #ifdef _DEBUG
                    assert(allocRef && rangeData);
                    #endif // _DEBUG
                    allocRef->multi_free_addr(numAllocs,rangeData,rangeData+numAllocs);
                }

            private:
                core::HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,StreamingGPUBufferAllocator,CPUAllocator>*    allocRef;
                size_type                                                                                                           numAllocs;
                size_type*                                                                                                          rangeData;
        };
        core::GPUEventDeferredHandlerST<DeferredFreeFunctor> deferredFrees;
};

}
}

#endif // __IRR_STREAMING_TRANSIENT_DATA_BUFFER_H__



