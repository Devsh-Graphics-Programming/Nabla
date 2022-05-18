#include "nbl/video/alloc/StreamingGPUBufferAllocator.h"

#include "nbl/video/ILogicalDevice.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl::video
{

void* StreamingGPUBufferAllocator::mapWrapper(IDriverMemoryAllocation* mem, core::bitflag<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAGS> access, const IDriverMemoryAllocation::MemoryRange& range) noexcept
{
    IDriverMemoryAllocation::MappedMemoryRange memory(mem,range.offset,range.length);
    return mDriver->mapMemory(memory, access);
}

void StreamingGPUBufferAllocator::unmapWrapper(IDriverMemoryAllocation* mem) noexcept
{
    mDriver->unmapMemory(mem);
}

}