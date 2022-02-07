// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_VIDEO_STREAMING_GPUBUFFER_ALLOCATOR_H__
#define __NBL_VIDEO_STREAMING_GPUBUFFER_ALLOCATOR_H__

#include "nbl/video/alloc/SimpleGPUBufferAllocator.h"

namespace nbl
{
namespace video
{
class StreamingGPUBufferAllocator : protected SimpleGPUBufferAllocator
{
protected:
    inline uint8_t* mapWholeBuffer(IGPUBuffer* buff) noexcept
    {
        auto rangeToMap = IDriverMemoryAllocation::MemoryRange{0u, buff->getSize()};
        auto memory = const_cast<IDriverMemoryAllocation*>(buff->getBoundMemory());
        auto mappingCaps = memory->getMappingCaps() & IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;
        return reinterpret_cast<uint8_t*>(memory->mapMemoryRange(static_cast<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG>(mappingCaps), rangeToMap));
    }

public:
    typedef std::pair<IGPUBuffer*, uint8_t*> value_type;

    StreamingGPUBufferAllocator(IDriver* inDriver, const IDriverMemoryBacked::SDriverMemoryRequirements& bufferReqs)
        : SimpleGPUBufferAllocator(inDriver, bufferReqs)
    {
        assert(mBufferMemReqs.mappingCapability & IDriverMemoryAllocation::EMCAF_READ_AND_WRITE);  // have to have mapping access to the buffer!
    }

    inline value_type allocate(size_t bytes, size_t alignment) noexcept
    {
        auto buff = SimpleGPUBufferAllocator::allocate(bytes, alignment);
        if(!buff)
            return {nullptr, nullptr};
        auto mappedPtr = mapWholeBuffer(buff);
        if(!mappedPtr)
        {
            SimpleGPUBufferAllocator::deallocate(buff);
            return {nullptr, nullptr};
        }
        return {buff, mappedPtr};
    }

    template<class AddressAllocator>
    inline void reallocate(value_type& allocation, size_t bytes, size_t alignment, const AddressAllocator& allocToQueryOffsets, bool copyBuffers = true) noexcept
    {
        auto newAlloc = allocate(bytes, alignment);
        if(!newAlloc.first)
        {
            deallocate(allocation);
            return;
        }

        //move contents
        if(copyBuffers)
        {
            auto oldOffset_copyRange = getOldOffset_CopyRange_OldSize(allocation, bytes, allocToQueryOffsets);

            if(allocation.second && (allocation.first->getBoundMemory()->getCurrentMappingCaps() & IDriverMemoryAllocation::EMCAF_READ))  // can read from old
                memcpy(newAlloc.second, allocation.second + oldOffset_copyRange.first, oldOffset_copyRange.second);
            else
                copyBufferWrapper(allocation.first, newAlloc.first, oldOffset_copyRange.first, 0u, oldOffset_copyRange.second);
        }

        //swap the internals of buffers and book keeping
        const_cast<IDriverMemoryAllocation*>(allocation.first->getBoundMemory())->unmapMemory();
        allocation.first->pseudoMoveAssign(newAlloc.first);
        newAlloc.first->drop();
        allocation.second = newAlloc.second;
    }

    inline void deallocate(value_type& allocation) noexcept
    {
        allocation.second = nullptr;
        const_cast<IDriverMemoryAllocation*>(allocation.first->getBoundMemory())->unmapMemory();
        SimpleGPUBufferAllocator::deallocate(allocation.first);
    }

    //to expose base functions again
    IDriver* getDriver() noexcept { return SimpleGPUBufferAllocator::getDriver(); }
};

}
}

#endif
