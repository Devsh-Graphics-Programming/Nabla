// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_SIMPLE_GPU_BUFFER_ALLOCATOR_H__
#define __NBL_VIDEO_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

#include "nbl/core/alloc/address_allocator_traits.h"
#include "nbl/video/alloc/GPUMemoryAllocatorBase.h"

#include "nbl/video/IGPUBuffer.h"

namespace nbl::video
{

//! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
//!     Vulkan and OpenGL should have different implementations
//!     We should probably split GPUBuffer from Memory allocation (fake it on OpenGL)
class SimpleGPUBufferAllocator : public GPUMemoryAllocatorBase
{
    protected:
        IDriverMemoryBacked::SDriverMemoryRequirements  mBufferMemReqs;
        bool canUpdateViaCmdBuff;

    public:
        using value_type = core::smart_refctd_ptr<IGPUBuffer>;

        SimpleGPUBufferAllocator(ILogicalDevice* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs, bool _canUpdateViaCmdBuff=false) :
                        GPUMemoryAllocatorBase(inDriver), mBufferMemReqs(bufferReqs), canUpdateViaCmdBuff(_canUpdateViaCmdBuff)
        {
        }
        virtual ~SimpleGPUBufferAllocator() = default;

        value_type  allocate(size_t bytes, size_t alignment) noexcept;

        inline void deallocate(value_type& allocation) noexcept
        {
            //commented out, it's not fulfilling its purpose since we use smart refctd ptr
            //assert(allocation->getReferenceCount()==1);
            allocation = nullptr;
        }
};

}

#endif

