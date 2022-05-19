// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_C_SIMPLE_BUFFER_ALLOCATOR_H__
#define __NBL_VIDEO_C_SIMPLE_BUFFER_ALLOCATOR_H__

#include "nbl/core/alloc/address_allocator_traits.h"
#include "nbl/video/alloc/IBufferAllocator.h"

#include "nbl/video/IGPUBuffer.h"

namespace nbl::video
{
class CSimpleBufferAllocator : public IBufferAllocator
{
    uint32_t m_memoryTypesToUse;

  public:
    using value_type = asset::SBufferBinding<IGPUBuffer>;

    CSimpleBufferAllocator(core::smart_refctd_ptr<ILogicalDevice>&& _device, const uint32_t _memoryTypesToUse) : IBufferAllocator(std::move(_device)), m_memoryTypesToUse(_memoryTypesToUse) {}
    virtual ~CSimpleBufferAllocator() = default;

    inline ILogicalDevice* getDevice() {return static_cast<ILogicalDevice*>(m_memoryAllocator.get());}

    value_type allocate(
        const IGPUBuffer::SCreationParams& creationParams,
        const core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags=IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE);

    void deallocate(value_type& allocation)
    {
        allocation = {IDriverMemoryAllocator::InvalidMemoryOffset,nullptr};
    }
};

}

#endif

