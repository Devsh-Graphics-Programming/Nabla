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
};


template< typename _size_type=uint32_t, class CPUAllocator=core::allocator<uint8_t> >
class StreamingTransientDataBufferST : protected core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor, public virtual core::IReferenceCounted
{
        _IRR_INTERFACE_CHILD_DEFAULT(StreamingTransientDataBufferST);
    protected:
        typedef core::GeneralpurposeAddressAllocator<_size_type>                                                    BasicAddressAllocator;
        HeterogenousMemoryAddressAllocatorAdaptor<BasicAddressAllocator,StreamingGPUBufferAllocator,CPUAllocator>   mAllocator; // no point for a streaming buffer to grow
        typedef core::address_allocator_traits<decltype(mAllocator)>                                                alloc_traits;
    public:
        typedef typename BasicAddressAllocator::size_type           size_type;
        static constexpr size_type                                  invalid_address = address_type_traits<size_type>::invalid_address;

        //!
        /**
        \param default minAllocSize is carefully picked to reflect the lowest nonCoherentAtomSize under Vulkan 1.1 which is not 1u .*/
        StreamingTransientDataBufferST(IVideoDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs,
                                       const CPUAllocator& reservedMemAllocator, size_type bufSz, size_type minAllocSize=64u) :
                                                mAllocator(reservedMemAllocator,StreamingGPUBufferAllocator(inDriver,bufferReqs),bufSz,minAllocSize)
        {
        }

        virtual ~StreamingTransientDataBufferST() {}


        inline bool         needsManualFlushOrInvalidate() const {return !(getBuffer()->getMemoryReqs().mappingCapability&video::IDriverMemoryAllocation::EMCF_COHERENT);}

        inline IGPUBuffer*  getBuffer() noexcept {return core::impl::FriendOfHeterogenousMemoryAddressAllocatorAdaptor(mAllocator).getAllocatedBuffer();}

        // have to try each fence once (need a function for that)
        // but only try a few fences before the next alloc attempt
        // hpw many fences to try at each step?
        inline size_type    multi_alloc() noexcept //! TODO:
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

        inline size_type    multi_place() noexcept //! TODO:
        {
        #ifdef _DEBUG
            assert(GPUBuffer has write mapping flags);
        #endif // _DEBUG
            multi_alloc();
            // fill with data
            for
            {
                memcpy();
            }
        }

        inline void         multi_free(const IDriverMemoryAllocation::MemoryRange& allocation, IDriverFence* fence) noexcept //! TODO:
        {
            if (fence)
                deferredFrees.emplace_back(fence,allocation);
            else
                mAllocator.multi_free_addr(allocation.offset,allocation.length);
        }

        //! Returns if all fences have completed
        inline bool         execute_frees(size_t timeOutNs=0u)
        {
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

        // extras
        inline void         swap_fences(std::pair<const IDriverFence*,IDriverFence*>* swaplist_begin,
                                        std::pair<const IDriverFence*,IDriverFence*>* swaplist_end) noexcept
        {
            std::sort(deferredFrees.begin(),deferredFrees.end(),deferredFreeCompareFunc);

            for (auto it=swaplist_begin; it!=swaplist_end; it++)
            {
                auto lbound = std::lower_bound(deferredFrees.begin(),deferredFrees.end(),*it,deferredFreeCompareFunc);
                auto ubound = std::upper_bound(lbound,deferredFrees.end(),*it,deferredFreeCompareFunc);
                for (auto it2=lbound; it2!=ubound; it2++)
                {
                    it->second->grab();
                    auto oldFence = it2->first;
                    it2->first = it->second;
                    oldFence->drop();
                }
            }
        }
    protected:
        core::vector< std::pair<IDriverFence*,IDriverMemoryAllocation::MemoryRange> > deferredFrees;
        auto deferredFreeCompareFunc = [](const deferredFrees::value_type &lhs, const deferredFrees::value_type &rhs) {return lhs.first < rhs.first;};
};

}
}

#endif // __IRR_STREAMING_TRANSIENT_DATA_BUFFER_H__



