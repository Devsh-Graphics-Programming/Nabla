#ifndef __IRR_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__
#define __IRR_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__


#include "irr/video/GPUMemoryAllocatorBase.h"

namespace irr
{
namespace video
{

//! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
template<class HostAllocator = core::allocator<uint8_t> >
class HostDeviceMirrorBufferAllocator : public GPUMemoryAllocatorBase
{
        typedef std::pair<uint8_t*,IGPUBuffer*> AllocationType;

        HostAllocator                                            hostAllocator;
        AllocationType                                          lastAllocation;

        AllocationType createBuffers(size_t bytes, size_t alignment=_IRR_SIMD_ALIGNMENT);
    public:
        HostDeviceMirrorBufferAllocator(IVideoDriver* inDriver) : GPUMemoryAllocatorBase(inDriver), lastAllocation(nullptr,nullptr) {}
        virtual ~HostDeviceMirrorBufferAllocator()
        {
        #ifdef _DEBUG
            assert(!lastAllocation.first && !lastAllocation.second);
        #endif // _DEBUG
        }

        inline void*        allocate(size_t bytes) noexcept
        {
        #ifdef _DEBUG
            assert(bytes && !lastAllocation.first && !lastAllocation.second);
        #endif // _DEBUG
            lastAllocation = createBuffers(bytes);
            return lastAllocation.first;
        }

        template<class AddressAllocator>
        inline void*        reallocate(void* addr, size_t bytes, const AddressAllocator& allocToQueryOffsets) noexcept
        {
        #ifdef _DEBUG
            assert(addr && lastAllocation.first==addr && lastAllocation.second);
        #endif // _DEBUG

            auto alignment = core::address_allocator_traits<AddressAllocator>::max_alignment(allocToQueryOffsets);
            auto oldMemReqs = lastAllocation.second->getMemoryReqs();
            auto tmp = createBuffers(bytes,alignment);

            //move contents
            size_t oldOffset = core::address_allocator_traits<AddressAllocator>::get_align_offset(allocToQueryOffsets);
            auto copyRangeLen = std::min(oldMemReqs.vulkanReqs.size-oldOffset,bytes);


            memcpy(tmp.first,lastAllocation.first+oldOffset,copyRangeLen);
            hostAllocator.deallocate(lastAllocation.first,oldMemReqs.vulkanReqs.size);

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
            assert(addr && lastAllocation.first==addr && lastAllocation.second);
        #endif // _DEBUG
            hostAllocator.deallocate(lastAllocation.first,lastAllocation.second->getMemoryReqs().vulkanReqs.size);
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

#include "IVideoDriver.h"

namespace irr
{
namespace video
{

template<class HostAllocator>
inline typename HostDeviceMirrorBufferAllocator<HostAllocator>::AllocationType HostDeviceMirrorBufferAllocator<HostAllocator>::createBuffers(size_t bytes, size_t alignment)
{
    AllocationType retval;
    retval.first = hostAllocator.allocate(bytes,alignment);

    auto memReqs = mDriver->getDeviceLocalGPUMemoryReqs();
    memReqs.vulkanReqs.size = bytes;
    retval.second = mDriver->createGPUBufferOnDedMem(memReqs,false);

    return retval;
}
}
}

#endif // __IRR_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__
