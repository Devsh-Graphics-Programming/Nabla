// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/alloc/CSimpleBufferAllocator.h"

using namespace nbl;
using namespace video;

CSimpleBufferAllocator::value_type CSimpleBufferAllocator::allocate(
    const IGPUBuffer::SCreationParams& creationParams,
    const core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags=IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE)
{
    auto buffer = getDevice()->createBuffer(bufferParams);
    auto reqs = buffer->getMemoryReqs2();
    reqs.memoryTypeBits &= m_memoryTypesToUse;
    auto mem = m_memoryAllocator->allocate(reqs,buffer.get(),allocateFlags);
    if (!mem)
        return {nullptr,0xdeadbeefull};
    return {std::move(buffer),0ull};
}