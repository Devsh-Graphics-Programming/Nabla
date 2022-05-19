// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_STREAMING_GPUBUFFER_ALLOCATOR_H__
#define __NBL_VIDEO_STREAMING_GPUBUFFER_ALLOCATOR_H__

#include "nbl/video/alloc/CSimpleBufferAllocator.h"

namespace nbl::video
{
class CStreamingBufferAllocator : protected CSimpleBufferAllocator
{
public:
    struct value_type
    {
        typename CSimpleBufferAllocator::value_type bufferBinding;
        uint8_t* ptr;
    };

    using CSimpleBufferAllocator::CSimpleBufferAllocator;
    virtual ~CStreamingBufferAllocator() = default;

    value_type allocate(const IGPUBuffer::SCreationParams& creationParams, const core::bitflag<IDriverMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags = IDriverMemoryAllocation::EMAF_NONE)
    {
        auto bufferBinding = CSimpleBufferAllocator::allocate(creationParams, allocateFlags);
        uint8_t* mappedPtr = nullptr;
        if (bufferBinding.buffer)
        {
            IDriverMemoryAllocation* mem = bufferBinding.buffer->getBoundMemory();
            if (mem->isCurrentlyMapped())
            {
                assert(mem->getMappedRange().offset == 0ull && mem->getMappedRange().length == mem->getAllocationSize());
                mappedPtr = reinterpret_cast<uint8_t*>(mem->getMappedPointer());
            }
            else
            {
                core::bitflag<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> access(IDriverMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);
                const auto memProps = mem->getMemoryPropertyFlags();
                if (memProps.hasFlags(IDriverMemoryAllocation::EMPF_HOST_READABLE_BIT))
                    access |= IDriverMemoryAllocation::EMCAF_READ;
                if (memProps.hasFlags(IDriverMemoryAllocation::EMPF_HOST_WRITABLE_BIT))
                    access |= IDriverMemoryAllocation::EMCAF_WRITE;
                assert(access.value);
                IDriverMemoryAllocation::MappedMemoryRange memoryRange = {mem,0ull,mem->getAllocationSize()};
                mappedPtr = reinterpret_cast<uint8_t*>(getDevice()->mapMemory(memoryRange, access));
            }
            if (!mappedPtr)
                CSimpleBufferAllocator::deallocate(bufferBinding);
            mappedPtr += bufferBinding.buffer->getBoundMemoryOffset() + bufferBinding.offset;
        }
        return {std::move(bufferBinding),mappedPtr};
    }

    inline void deallocate(value_type& allocation)
    {
        allocation.ptr = nullptr;
        auto* mem = allocation.bufferBinding.buffer->getBoundMemory();
        if (mem->getReferenceCount() == 1)
            getDevice()->unmapMemory(mem);
        CSimpleBufferAllocator::deallocate(allocation.bufferBinding);
    }
};
}

#endif
