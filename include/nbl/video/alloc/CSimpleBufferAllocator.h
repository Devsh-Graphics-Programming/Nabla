// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_C_SIMPLE_BUFFER_ALLOCATOR_H_
#define _NBL_VIDEO_C_SIMPLE_BUFFER_ALLOCATOR_H_

#include "nbl/video/IDeviceMemoryAllocator.h"
#include "nbl/video/alloc/IBufferAllocator.h"

namespace nbl::video
{

class CSimpleBufferAllocator : public IBufferAllocator
{
    core::smart_refctd_ptr<ILogicalDevice> m_device;
    uint32_t m_memoryTypesToUse;

  public:
    using value_type = asset::SBufferBinding<IGPUBuffer>;

    CSimpleBufferAllocator(core::smart_refctd_ptr<ILogicalDevice>&& _device, const uint32_t _memoryTypesToUse) : m_device(std::move(_device)), m_memoryTypesToUse(_memoryTypesToUse) {}
    virtual ~CSimpleBufferAllocator() = default;

    inline ILogicalDevice* getDevice() {return m_device.get();}

    value_type allocate(
        IGPUBuffer::SCreationParams&& creationParams,
        const core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags=IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE
    );

    inline void deallocate(value_type& allocation)
    {
        allocation = {IDeviceMemoryAllocator::InvalidMemoryOffset,nullptr};
    }
};

}

#endif

