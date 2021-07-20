// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_SIMPLE_GPU_BUFFER_ALLOCATOR_H__
#define __NBL_VIDEO_SIMPLE_GPU_BUFFER_ALLOCATOR_H__

#include "nbl/core/alloc/address_allocator_traits.h"
#include "nbl/video/alloc/GPUMemoryAllocatorBase.h"

namespace nbl::video
{

//! TODO: Use a GPU heap allocator instead of buffer directly -- after move to Vulkan only
//!     reallocate should hand out a new buffer/allocation
//!     Vulkan and OpenGL should have different implementations
//!     We should probably split GPUBuffer from Memory allocation (fake it on OpenGL)
class SimpleGPUBufferAllocator : public GPUMemoryAllocatorBase
{
    protected:
        IDriverMemoryBacked::SDriverMemoryRequirements  mBufferMemReqs;

        template<class AddressAllocator>
        std::tuple<typename AddressAllocator::size_type,size_t,size_t> getOldOffset_CopyRange_OldSize(IGPUBuffer* oldBuff, size_t bytes, const AddressAllocator& allocToQueryOffsets)
        {
            auto oldSize = oldBuff->getSize();
            auto oldOffset = core::address_allocator_traits<AddressAllocator>::get_combined_offset(allocToQueryOffsets);
            auto copyRangeLen = core::min<size_t>(oldSize-oldOffset,bytes);
            return std::make_tuple(oldOffset,copyRangeLen,oldSize);
        }
    public:
        typedef IGPUBuffer* value_type;

        SimpleGPUBufferAllocator(ILogicalDevice* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs) :
                        GPUMemoryAllocatorBase(inDriver), mBufferMemReqs(bufferReqs)
        {
        }

        value_type  allocate(size_t bytes, size_t alignment) noexcept;

#if 0
        template<class AddressAllocator>
        inline void             reallocate(value_type& allocation, size_t bytes, size_t alignment, const AddressAllocator& allocToQueryOffsets, bool copyBuffers=true) noexcept
        {
            auto tmp = allocate(bytes,alignment);
            if (!tmp)
            {
                deallocate(allocation);
                return;
            }

            //move contents
            if (copyBuffers)
            {
                auto oldOffset_copyRange = getOldOffset_CopyRange_OldSize(allocation,bytes,allocToQueryOffsets);
                copyBufferWrapper(allocation,tmp,oldOffset_copyRange.first,0u,oldOffset_copyRange.second);
            }

            //swap the internals of buffers
            allocation->pseudoMoveAssign(tmp);
            tmp->drop();
        }
#endif

        inline void             deallocate(value_type& allocation) noexcept
        {
            allocation->drop();
            allocation = nullptr;
        }
};

}

#endif

