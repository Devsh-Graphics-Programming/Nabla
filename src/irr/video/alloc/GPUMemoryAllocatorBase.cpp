#include "irr/video/alloc/GPUMemoryAllocatorBase.h"

#include "IVideoDriver.h"

using namespace irr;
using namespace video;


void            GPUMemoryAllocatorBase::copyBuffersWrapper(IGPUBuffer* oldBuffer, IGPUBuffer* newBuffer, size_t oldOffset, size_t newOffset, size_t copyRangeLen)
{
    mDriver->copyBuffer(oldBuffer,newBuffer,oldOffset,newOffset,copyRangeLen);
}
