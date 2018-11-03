#ifndef __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__
#define __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

#include "IGPUBuffer.h"
#include "irr/video/GPUMemoryAllocatorBase.h"


namespace irr
{
namespace video
{

namespace impl
{
    class SimpleGPUBufferAllocatorBase : public GPUMemoryAllocatorBase
    {
        protected:
            IGPUBuffer* createGPUBuffer(const IDriverMemoryBacked::SDriverMemoryRequirements& bufferMemReqs);
        public:
            using GPUMemoryAllocatorBase::GPUMemoryAllocatorBase;
    };
}

//! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
template<size_t kNumBuffers> // this is like a GPU memory RAID1 :)
class SimpleGPUBufferAllocator : public impl::SimpleGPUBufferAllocatorBase
{
        VkDeviceSize                                    mSizeOverride;
        VkDeviceSize                                    mAlignmentOverride; /// Used and valid only in Vulkan
        IDriverMemoryBacked::SDriverMemoryRequirements  mBufferMemReqs[kNumBuffers];
        std::pair<uint8_t*,IGPUBuffer*[kNumBuffers]>    lastAllocation;

        inline decltype(lastAllocation) createAndMapBuffers()
        {
            decltype(lastAllocation) retval; retval.first = nullptr;
            for (size_t i=0; i<kNumBuffers; i++)
            {
                mBufferMemReqs[i].vulkanReqs.size = mSizeOverride;
                mBufferMemReqs[i].vulkanReqs.alignment = mAlignmentOverride;
                retval.second[i] = createGPUBuffer(mBufferMemReqs[i]);
            }

            // only try to map the first buffer
            {
                auto mappingCaps = mBufferMemReqs[0u].mappingCapability&IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;
                if (mappingCaps)
                {
                    auto rangeToMap = IDriverMemoryAllocation::MemoryRange{0u,mSizeOverride};
                    auto memory = const_cast<IDriverMemoryAllocation*>(retval.second[0u]->getBoundMemory());
                    retval.first  = reinterpret_cast<uint8_t*>(memory->mapMemoryRange(static_cast<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG>(mappingCaps),rangeToMap));
                }
            }
            return retval;
        }
    public:
        SimpleGPUBufferAllocator(IVideoDriver* inDriver, VkDeviceSize allBufferSize, VkDeviceSize allBufferAlign, const IDriverMemoryBacked::SDriverMemoryRequirements* bufferReqs) :
                impl::SimpleGPUBufferAllocatorBase(inDriver), mSizeOverride(allBufferSize), mAlignmentOverride(allBufferAlign)
        {
            memcpy(mBufferMemReqs,bufferReqs,kNumBuffers*sizeof(IDriverMemoryBacked::SDriverMemoryRequirements));
            lastAllocation.first = nullptr;
            for (size_t i=0; i<kNumBuffers; i++)
                lastAllocation.second[i] = nullptr;
        }

        inline void*        allocate(size_t bytes) noexcept
        {
        #ifdef _DEBUG
            assert(!lastAllocation.first);
            for (size_t i=0; i<kNumBuffers; i++)
                assert(!lastAllocation.second[i]);
        #endif // _DEBUG
            mSizeOverride = bytes;
            lastAllocation = createAndMapBuffers();
            return lastAllocation.first;
        }

        template<class AddressAllocator>
        inline void*        reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets, const bool* copyBuffers=nullptr) noexcept
        {
        #ifdef _DEBUG
            assert(lastAllocation.first==reinterpret_cast<uint8_t*>(addr));
            for (size_t i=0; i<kNumBuffers; i++)
                assert(lastAllocation.second[i]);
        #endif // _DEBUG

            // set up new size
            auto oldSize = mSizeOverride;
            mSizeOverride = bytes;
            //allocate new buffer
            auto tmp = createAndMapBuffers();

            //move contents
            for (size_t i=0; i<kNumBuffers; i++)
            {
                if (copyBuffers&&!copyBuffers[i])
                    continue;

                bool firstBuffer = i==0u;

                // only first buffer is bound to allocator
                size_t oldOffset = firstBuffer ? allocToQueryOffsets.get_align_offset():0u;
                size_t newOffset = firstBuffer ? AddressAllocator::aligned_start_offset(reinterpret_cast<size_t>(tmp.first),allocToQueryOffsets.max_alignment()):0u;
                auto copyRangeLen = std::min(oldSize-oldOffset,bytes-newOffset);

                if (firstBuffer && lastAllocation.first && tmp.first && // non-null buffers
                    (lastAllocation.second[0u]->getBoundMemory()->getCurrentMappingCaps()&IDriverMemoryAllocation::EMCAF_READ) &&
                    (tmp.second[0u]->getBoundMemory()->getCurrentMappingCaps()&IDriverMemoryAllocation::EMCAF_WRITE))
                {
                    memcpy(tmp.first+newOffset,lastAllocation.first+oldOffset,copyRangeLen);
                }
                else
                    copyBuffersWrapper(lastAllocation.second[i],tmp.second[i],oldOffset,newOffset,copyRangeLen);
            }

            //swap the internals of buffers
            if (lastAllocation.first)
                const_cast<IDriverMemoryAllocation*>(lastAllocation[0u].second->getBoundMemory())->unmapMemory();
            for (size_t i=0; i<kNumBuffers; i++)
            {
                lastAllocation.second[i]->pseudoMoveAssign(tmp.second[i]);
                tmp.second[i]->drop();
            }

            //book-keeping and return
            lastAllocation.first = tmp.first;
            return lastAllocation.first;
        }

        inline void         deallocate(void* addr) noexcept
        {
        #ifdef _DEBUG
            assert(lastAllocation.first==reinterpret_cast<uint8_t*>(addr));
            for (size_t i=0; i<kNumBuffers; i++)
                assert(lastAllocation.second[i]);
        #endif // _DEBUG

            if (lastAllocation.first)
                const_cast<IDriverMemoryAllocation*>(lastAllocation[0u].second->getBoundMemory())->unmapMemory();
            lastAllocation.first = nullptr;
            for (size_t i=0; i<kNumBuffers; i++)
            {
                lastAllocation.second[i]->drop();
                lastAllocation.second[i] = nullptr;
            }
        }

/*
        // extras
        inline IGPUBuffer*  getAllocatedBuffer(size_t ix)
        {
        #ifdef _DEBUG
            assert(ix<kNumBuffers);
        #endif // _DEBUG
            return lastAllocation.second[ix];
        }
*/
};

}
}


#endif // __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

