#include <cstring>

#include "irr/video/SimpleGPUBufferAllocator.h"
#include "IVideoDriver.h"


using namespace irr;
using namespace video;

IGPUBuffer* video::impl::SimpleGPUBufferAllocatorBase::createGPUBuffer(const IDriverMemoryBacked::SDriverMemoryRequirements& bufferMemReqs)
{
    return mDriver->createGPUBufferOnDedMem(bufferMemReqs,false);
}
