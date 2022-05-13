#include "nbl/video/alloc/StreamingGPUBufferAllocator.h"

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

void* StreamingGPUBufferAllocator::mapWrapper(IDriverMemoryAllocation* mem, const IDriverMemoryAllocation::MemoryRange& range) noexcept
{
    IDriverMemoryAllocation::MappedMemoryRange memory(mem,range.offset,range.length);
    return mDriver->mapMemory(memory);
}

void StreamingGPUBufferAllocator::unmapWrapper(IDriverMemoryAllocation* mem) noexcept
{
    mDriver->unmapMemory(mem);
}

}