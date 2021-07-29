// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/alloc/SimpleGPUBufferAllocator.h"


using namespace nbl;
using namespace video;


SimpleGPUBufferAllocator::value_type SimpleGPUBufferAllocator::allocate(size_t bytes, size_t alignment) noexcept
{
    auto reqs = mBufferMemReqs;
    reqs.vulkanReqs.size = bytes;
    reqs.vulkanReqs.alignment = alignment;
    IGPUBuffer* bufptr = nullptr;
    {
        auto buff = mDriver->createGPUBufferOnDedMem(reqs, false);
        bufptr = buff.get();
        bufptr->grab(); // don't want to be passing smart pointers around, this allocator it the only owner!
    }
    return bufptr;
}
