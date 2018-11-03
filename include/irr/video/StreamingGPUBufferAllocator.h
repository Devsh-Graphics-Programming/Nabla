#ifndef __IRR_STREAMING_GPUBUFFER_ALLOCATOR_H__
#define __IRR_STREAMING_GPUBUFFER_ALLOCATOR_H__

#include "IGPUBuffer.h"
#include "irr/video/GPUMemoryAllocatorBase.h"

namespace irr
{
namespace video
{

class IVideoDriver;

//! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
class StreamingGPUBufferAllocator : public GPUMemoryAllocatorBase
{
        IDriverMemoryBacked::SDriverMemoryRequirements  mBufferMemReqs;
        std::pair<uint8_t*,IGPUBuffer*>                 lastAllocation;

        decltype(lastAllocation)    createAndMapBuffer();
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
        inline void*        reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets, bool copyBuffers=true) noexcept
        {
        #ifdef _DEBUG
            assert(lastAllocation.first && lastAllocation.second);
        #endif // _DEBUG

            // set up new size
            auto oldSize = mBufferMemReqs.vulkanReqs.size;
            mBufferMemReqs.vulkanReqs.size = bytes;
            //allocate new buffer
            auto tmp = createAndMapBuffer();
            auto newOffset = AddressAllocator::aligned_start_offset(reinterpret_cast<size_t>(tmp.first),allocToQueryOffsets.max_alignment());

            //move contents
            if (copyBuffers)
            {
                // only first buffer is bound to allocator
                auto oldOffset = allocToQueryOffsets.get_align_offset();
                auto copyRangeLen = std::min(oldSize-oldOffset,bytes-newOffset);

                if ((lastAllocation.second->getBoundMemory()->getCurrentMappingCaps()&IDriverMemoryAllocation::EMCAF_READ) &&
                    (tmp.second->getBoundMemory()->getCurrentMappingCaps()&IDriverMemoryAllocation::EMCAF_WRITE)) // can read from old and write to new
                {
                    memcpy(tmp.first+newOffset,lastAllocation.first+oldOffset,copyRangeLen);
                }
                else
                    copyBufferWrapper(lastAllocation.second,tmp.second,oldOffset,newOffset,copyRangeLen);
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

}
}

#endif // __IRR_STREAMING_GPUBUFFER_ALLOCATOR_H__
