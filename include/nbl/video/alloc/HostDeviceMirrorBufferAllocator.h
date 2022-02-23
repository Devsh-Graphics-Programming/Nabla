// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__
#define __NBL_VIDEO_HOST_DEVICE_MIRROR_BUFFER_ALLOCATOR_H__


#include "nbl/video/alloc/SimpleGPUBufferAllocator.h"

namespace nbl::video
{

//class ILogicalDevice;

template<class HostAllocator = core::allocator<uint8_t> >
class HostDeviceMirrorBufferAllocator : protected SimpleGPUBufferAllocator
{
        HostAllocator hostAllocator;
    public:
        struct value_type
        {
            typename SimpleGPUBufferAllocator::value_type buffer;
            uint8_t* ptr; // maybe a ICPUBuffer in the future?
        };

        HostDeviceMirrorBufferAllocator(ILogicalDevice* inDriver);
        virtual ~HostDeviceMirrorBufferAllocator() = default;

        inline value_type   allocate(size_t bytes, size_t alignment) noexcept
        {
            auto buff =  SimpleGPUBufferAllocator::allocate(bytes,alignment);
            if (!buff)
                return {nullptr,nullptr};
            auto hostPtr = hostAllocator.allocate(bytes,alignment);
            if (!hostPtr)
            {
                SimpleGPUBufferAllocator::deallocate(buff);
                return {nullptr,nullptr};
            }
            return {std::move(buff),hostPtr};
        }

        inline void         deallocate(value_type& allocation) noexcept
        {
            hostAllocator.deallocate(allocation.ptr,allocation.buffer->getSize());
            SimpleGPUBufferAllocator::deallocate(allocation.buffer);
            allocation.ptr = nullptr;
        }
#if 0
        //to expose base functions again
        IDriver*   getDriver() noexcept {return SimpleGPUBufferAllocator::getDriver();}
#endif
};


}

#include "nbl/video/ILogicalDevice.h"

namespace nbl::video
{

template<class HostAllocator>
HostDeviceMirrorBufferAllocator<HostAllocator>::HostDeviceMirrorBufferAllocator(ILogicalDevice* inDriver) : SimpleGPUBufferAllocator(inDriver,inDriver->getDeviceLocalGPUMemoryReqs()) {}

}

#endif
