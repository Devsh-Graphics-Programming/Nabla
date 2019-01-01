#ifndef __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__
#define __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

#include "IGPUBuffer.h"
#include "irr/video/GPUMemoryAllocatorBase.h"

namespace irr
{
namespace video
{

    class IVideoDriver;

    //! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
    class SimpleGPUBufferAllocator : public GPUMemoryAllocatorBase
    {
        protected:
            IDriverMemoryBacked::SDriverMemoryRequirements  mBufferMemReqs;

            IGPUBuffer* createBuffer();
        public:
            SimpleGPUBufferAllocator(IVideoDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs) :
                            GPUMemoryAllocatorBase(inDriver), mBufferMemReqs(bufferReqs)
            {
            }

            inline IGPUBuffer*  allocate(size_t bytes) noexcept
            {
                mBufferMemReqs.vulkanReqs.size = bytes;
                return createBuffer();
            }

            template<class AddressAllocator>
            inline void             reallocate(IGPUBuffer* buff, size_t bytes, const AddressAllocator& allocToQueryOffsets, bool copyBuffers=true) noexcept
            {
                // set up new size
                auto oldSize = mBufferMemReqs.vulkanReqs.size;
                mBufferMemReqs.vulkanReqs.size = bytes;
                //allocate new buffer
                auto tmp = createBuffer();

                //move contents
                if (copyBuffers)
                {
                    // only first buffer is bound to allocator
                    auto oldOffset = allocToQueryOffsets.get_align_offset();
                    auto copyRangeLen = std::min(oldSize-oldOffset,bytes);
                    copyBufferWrapper(buff,tmp,oldOffset,0u,copyRangeLen);
                }

                //swap the internals of buffers
                buff->pseudoMoveAssign(tmp);
                tmp->drop();
            }

            inline void             deallocate(IGPUBuffer* buff) noexcept
            {
                buff->drop();
            }
    };
}
}

#endif // __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

