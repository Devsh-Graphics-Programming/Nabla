// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_GPU_MEMORY_ALLOCATOR_BASE_H__
#define __NBL_VIDEO_GPU_MEMORY_ALLOCATOR_BASE_H__

namespace nbl::video
{
class ILogicalDevice;

class GPUMemoryAllocatorBase
{
protected:
    ILogicalDevice* mDriver;  // TODO: change to smartpointer backlink (after declarations_and_definitions branch merge)

    GPUMemoryAllocatorBase(ILogicalDevice* inDriver)
        : mDriver(inDriver) {}
    virtual ~GPUMemoryAllocatorBase() = default;

public:
    ILogicalDevice* getDriver() noexcept { return mDriver; }
};

}

#endif
