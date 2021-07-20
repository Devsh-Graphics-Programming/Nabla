// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_STREAMING_GPUBUFFER_ALLOCATOR_H__
#define __NBL_VIDEO_STREAMING_GPUBUFFER_ALLOCATOR_H__

#include "nbl/video/alloc/SimpleGPUBufferAllocator.h"

namespace nbl::video
{

class ILogicalDevice;

class StreamingGPUBufferAllocator : protected SimpleGPUBufferAllocator
{
    private:
        void* mapWrapper(IDriverMemoryAllocation* mem, IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG access, const IDriverMemoryAllocation::MemoryRange& range) noexcept;
        void unmapWrapper(IDriverMemoryAllocation* mem) noexcept;

    public:
        struct value_type
        {
            typename SimpleGPUBufferAllocator::value_type buffer;
            uint8_t* ptr;
        };

        StreamingGPUBufferAllocator(ILogicalDevice* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs) : SimpleGPUBufferAllocator(inDriver,bufferReqs)
        {
            assert(mBufferMemReqs.mappingCapability&IDriverMemoryAllocation::EMCAF_READ_AND_WRITE); // have to have mapping access to the buffer!
        }
        virtual ~StreamingGPUBufferAllocator() = default;

        inline value_type   allocate(size_t bytes, size_t alignment) noexcept
        {
            auto buff =  SimpleGPUBufferAllocator::allocate(bytes,alignment);
            if (!buff)
                return {nullptr,nullptr};
            auto* const mem = buff->getBoundMemory();
            uint8_t* mappedPtr;
            if (mem->isCurrentlyMapped())
            {
                assert(mem->getMappedRange().offset==0ull && mem->getMappedRange().length==mem->getAllocationSize()); // whole range must be mapped always
                mappedPtr = reinterpret_cast<uint8_t*>(mem->getMappedPointer());
            }
            else
            {
                const auto mappingCaps = mem->getMappingCaps()&IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;
                const auto rangeToMap = IDriverMemoryAllocation::MemoryRange{0u,mem->getAllocationSize()};
                mappedPtr = reinterpret_cast<uint8_t*>(mapWrapper(mem,static_cast<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG>(mappingCaps),rangeToMap));
            }
            if (!mappedPtr)
            {
                SimpleGPUBufferAllocator::deallocate(buff);
                return {nullptr,nullptr};
            }
            return {std::move(buff),mappedPtr+buff->getBoundMemoryOffset()};
        }

        inline void                 deallocate(value_type& allocation) noexcept
        {
            allocation.ptr = nullptr;
            auto* mem = allocation.buffer->getBoundMemory();
            if (mem->getReferenceCount()==1)
                unmapWrapper(mem);
            SimpleGPUBufferAllocator::deallocate(allocation.buffer);
        }
#if 0
        //to expose base functions again
        ILogicalDevice*   getDriver() noexcept {return SimpleGPUBufferAllocator::getDriver();}
#endif
};


}

#endif
