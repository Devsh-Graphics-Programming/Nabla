#include <cstring>

#include "irr/video/StreamingGPUBufferAllocator.h"
#include "IVideoDriver.h"


using namespace irr;
using namespace video;


decltype(StreamingGPUBufferAllocator::lastAllocation)   StreamingGPUBufferAllocator::createAndMapBuffer()
{
    decltype(lastAllocation) retval; retval.first = nullptr;
    retval.second = mDriver->createGPUBufferOnDedMem(mBufferMemReqs,false);

    auto rangeToMap = IDriverMemoryAllocation::MemoryRange{0u,mBufferMemReqs.vulkanReqs.size};
    auto memory = const_cast<IDriverMemoryAllocation*>(retval.second->getBoundMemory());
    auto mappingCaps = mBufferMemReqs.mappingCapability&IDriverMemoryAllocation::EMCAF_READ_AND_WRITE;
    retval.first  = reinterpret_cast<uint8_t*>(memory->mapMemoryRange(static_cast<IDriverMemoryAllocation::E_MAPPING_CPU_ACCESS_FLAG>(mappingCaps),rangeToMap));

    return retval;
}
