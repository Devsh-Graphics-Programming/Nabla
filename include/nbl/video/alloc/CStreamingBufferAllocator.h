// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_VIDEO_STREAMING_GPUBUFFER_ALLOCATOR_H_
#define _NBL_VIDEO_STREAMING_GPUBUFFER_ALLOCATOR_H_

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

        inline value_type allocate(IGPUBuffer::SCreationParams&& creationParams, const core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocateFlags = IDeviceMemoryAllocation::EMAF_NONE)
        {
            auto bufferBinding = CSimpleBufferAllocator::allocate(std::move(creationParams),allocateFlags);
            uint8_t* mappedPtr = nullptr;
            if (bufferBinding.buffer)
            {
                IDeviceMemoryAllocation* mem = bufferBinding.buffer->getBoundMemory();
                if (mem->isCurrentlyMapped())
                {
                    assert(mem->getMappedRange().offset == 0ull && mem->getMappedRange().length == mem->getAllocationSize());
                    mappedPtr = reinterpret_cast<uint8_t*>(mem->getMappedPointer());
                }
                else
                {
                    core::bitflag<IDeviceMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> access(IDeviceMemoryAllocation::EMCAF_NO_MAPPING_ACCESS);
                    const auto memProps = mem->getMemoryPropertyFlags();
                    if (memProps.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_READABLE_BIT))
                        access |= IDeviceMemoryAllocation::EMCAF_READ;
                    if (memProps.hasFlags(IDeviceMemoryAllocation::EMPF_HOST_WRITABLE_BIT))
                        access |= IDeviceMemoryAllocation::EMCAF_WRITE;
                    assert(access.value);
                    IDeviceMemoryAllocation::MappedMemoryRange memoryRange = {mem,0ull,mem->getAllocationSize()};
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
