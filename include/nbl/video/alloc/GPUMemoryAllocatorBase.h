// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_GPU_MEMORY_ALLOCATOR_BASE_H__
#define __NBL_VIDEO_GPU_MEMORY_ALLOCATOR_BASE_H__

#include "IGPUBuffer.h"

namespace nbl::video
{

class ILogicalDevice;

class GPUMemoryAllocatorBase
{
    protected:
        ILogicalDevice* mDriver;
        // TODO: figure out if this needs to be some sort of a deferred thing with a cmbuffer/event or something
        void            copyBuffersWrapper(IGPUBuffer* oldBuffer, IGPUBuffer* newBuffer, size_t oldOffset, size_t newOffset, size_t copyRangeLen);

        GPUMemoryAllocatorBase(ILogicalDevice* inDriver) : mDriver(inDriver) {}
        virtual ~GPUMemoryAllocatorBase() {}
    public:
        ILogicalDevice*    getDriver() noexcept {return mDriver;}
};

}


#endif
