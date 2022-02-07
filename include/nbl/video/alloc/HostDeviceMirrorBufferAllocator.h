// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__
#define __NBL_VIDEO_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__

#include "nbl/video/alloc/SimpleGPUBufferAllocator.h"

namespace nbl
{
namespace video
{
template<class HostAllocator = core::allocator<uint8_t> >
class HostDeviceMirrorBufferAllocator : protected SimpleGPUBufferAllocator
{
    HostAllocator hostAllocator;

public:
    typedef std::pair<IGPUBuffer*, uint8_t*> value_type;

    HostDeviceMirrorBufferAllocator(IDriver* inDriver);
    virtual ~HostDeviceMirrorBufferAllocator()
    {
    }

    inline value_type allocate(size_t bytes, size_t alignment) noexcept
    {
        auto buff = SimpleGPUBufferAllocator::allocate(bytes, alignment);
        if(!buff)
            return {nullptr, nullptr};
        auto hostPtr = hostAllocator.allocate(bytes, alignment);
        if(!hostPtr)
        {
            SimpleGPUBufferAllocator::deallocate(buff);
            return {nullptr, nullptr};
        }
        return {buff, hostPtr};
    }

    template<class AddressAllocator>
    inline void reallocate(value_type& allocation, size_t bytes, size_t alignment, const AddressAllocator& allocToQueryOffsets) noexcept
    {
        auto newAlloc = allocate(bytes, alignment);
        if(!newAlloc.first)
        {
            deallocate(allocation);
            return;
        }

        //move contents
        auto oldOffset_copyRange_oldSize = getOldOffset_CopyRange_OldSize(allocation.first, bytes, allocToQueryOffsets);

        copyBuffersWrapper(allocation.first, newAlloc.first, std::get<0u>(oldOffset_copyRange_oldSize), 0u, std::get<1u>(oldOffset_copyRange_oldSize));

        memcpy(newAlloc.second, allocation.second + std::get<0u>(oldOffset_copyRange_oldSize), std::get<1u>(oldOffset_copyRange_oldSize));

        //swap the internals of buffers and book keeping
        hostAllocator.deallocate(allocation.second, std::get<2u>(oldOffset_copyRange_oldSize));
        allocation.first->pseudoMoveAssign(newAlloc.first);
        newAlloc.first->drop();
        allocation.second = newAlloc.second;
    }

    inline void deallocate(value_type& allocation) noexcept
    {
        hostAllocator.deallocate(allocation.second, allocation.first->getSize());
        SimpleGPUBufferAllocator::deallocate(allocation.first);
        allocation.second = nullptr;
    }

    //to expose base functions again
    IDriver* getDriver() noexcept { return SimpleGPUBufferAllocator::getDriver(); }
};

}
}

#include "IDriver.h"

namespace nbl
{
namespace video
{
template<class HostAllocator>
HostDeviceMirrorBufferAllocator<HostAllocator>::HostDeviceMirrorBufferAllocator(IDriver* inDriver)
    : SimpleGPUBufferAllocator(inDriver, inDriver->getDeviceLocalGPUMemoryReqs()) {}

}
}

#endif
