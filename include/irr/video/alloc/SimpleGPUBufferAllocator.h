#ifndef __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__
#define __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

#include "irr/core/alloc/address_allocator_traits.h"
#include "irr/video/alloc/GPUMemoryAllocatorBase.h"
#include "IGPUBuffer.h"

namespace irr
{
namespace video
{

    //! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
    class SimpleGPUBufferAllocator : public GPUMemoryAllocatorBase
    {
        protected:
            IDriverMemoryBacked::SDriverMemoryRequirements  mBufferMemReqs;

            template<class AddressAllocator>
            std::tuple<typename AddressAllocator::size_type,size_t,size_t> getOldOffset_CopyRange_OldSize(IGPUBuffer* oldBuff, size_t bytes, const AddressAllocator& allocToQueryOffsets)
            {
                auto oldSize = oldBuff->getSize();
                auto oldOffset = core::address_allocator_traits<AddressAllocator>::get_combined_offset(allocToQueryOffsets);
                auto copyRangeLen = std::min(oldSize-oldOffset,bytes);
                return std::make_tuple(oldOffset,copyRangeLen,oldSize);
            }
        public:
            typedef IGPUBuffer* value_type;

            SimpleGPUBufferAllocator(IDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs) :
                            GPUMemoryAllocatorBase(inDriver), mBufferMemReqs(bufferReqs)
            {
            }

            value_type  allocate(size_t bytes, size_t alignment) noexcept;

            template<class AddressAllocator>
            inline void             reallocate(value_type& allocation, size_t bytes, size_t alignment, const AddressAllocator& allocToQueryOffsets, bool copyBuffers=true) noexcept
            {
                auto tmp = allocate(bytes,alignment);
                if (!tmp)
                {
                    deallocate(allocation);
                    return;
                }

                //move contents
                if (copyBuffers)
                {
                    auto oldOffset_copyRange = getOldOffset_CopyRange_OldSize(allocation,bytes,allocToQueryOffsets);
                    copyBufferWrapper(allocation,tmp,oldOffset_copyRange.first,0u,oldOffset_copyRange.second);
                }

                //swap the internals of buffers
                allocation->pseudoMoveAssign(tmp);
                tmp->drop();
            }

            inline void             deallocate(value_type& allocation) noexcept
            {
                allocation->drop();
                allocation = nullptr;
            }
    };
}
}

#endif // __IRR_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

