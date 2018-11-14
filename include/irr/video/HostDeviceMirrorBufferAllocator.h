#ifndef __IRR_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__
#define __IRR_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__


#include "irr/core/alloc/MultiBufferingAllocatorBase.h"
#include "irr/core/alloc/ResizableHeterogenousMemoryAllocator.h"
#include "irr/video/GPUMemoryAllocatorBase.h"

namespace irr
{
namespace video
{

//! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
class HostDeviceMirrorBufferAllocator : public GPUMemoryAllocatorBase
{
        std::pair<void*,IGPUBuffer*>                 lastAllocation;

        decltype(lastAllocation) createBuffers(size_t bytes, size_t alignment=_IRR_SIMD_ALIGNMENT);
    public:
        HostDeviceMirrorBufferAllocator(IVideoDriver* inDriver) : GPUMemoryAllocatorBase(inDriver), lastAllocation(nullptr,nullptr) {}

        inline void*        allocate(size_t bytes) noexcept
        {
        #ifdef _DEBUG
            assert(!lastAllocation.first && !lastAllocation.second);
        #endif // _DEBUG
            lastAllocation = createBuffers(bytes);
            return lastAllocation.first;
        }

        template<class AddressAllocator>
        inline void*        reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets) noexcept
        {
        #ifdef _DEBUG
            assert(!addr && lastAllocation.first==addr && lastAllocation.second);
        #endif // _DEBUG

            auto alignment = core::address_allocator_traits<AddressAllocator>::max_alignment(allocToQueryOffsets);
            auto oldMemReqs = lastAllocation.second->getMemoryReqs();
            auto tmp = createBuffers(bytes,alignment);

            //move contents
            size_t oldOffset = core::address_allocator_traits<AddressAllocator>::get_align_offset(allocToQueryOffsets);
            auto copyRangeLen = std::min(oldMemReqs.vulkanReqs.size-oldOffset,bytes);

            memcpy(tmp.first,reinterpret_cast<uint8_t*>(lastAllocation.first)+oldOffset,copyRangeLen);
            _IRR_ALIGNED_FREE(lastAllocation.first);
            copyBuffersWrapper(lastAllocation.second,tmp.second,oldOffset,0u,copyRangeLen);
            //swap the internals of buffers
            lastAllocation.second->pseudoMoveAssign(tmp.second);
            tmp.second->drop();

            //book-keeping and return
            lastAllocation.first = tmp.first;
            return lastAllocation.first;
        }

        inline void         deallocate(void* addr) noexcept
        {
        #ifdef _DEBUG
            assert(!addr && lastAllocation.first==addr && lastAllocation.second);
        #endif // _DEBUG
            _IRR_ALIGNED_FREE(lastAllocation.first);
            lastAllocation.second->drop();
            lastAllocation.first = nullptr;
            lastAllocation.second = nullptr;
        }


        // extras
        inline void*        getCPUStagingAreaPtr()
        {
            return lastAllocation.first;
        }

        inline IGPUBuffer*  getGPUBuffer()
        {
            return lastAllocation.second;
        }
};


}
}

#endif // __IRR_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__
