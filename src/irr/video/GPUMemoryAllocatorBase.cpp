#include "irr/video/GPUMemoryAllocatorBase.h"

#include "IVideoDriver.h"

using namespace irr;
using namespace video;


void            GPUMemoryAllocatorBase::copyBuffersWrapper(IGPUBuffer* oldBuffer, IGPUBuffer* newBuffer, size_t oldOffset, size_t newOffset, size_t copyRangeLen)
{
    mDriver->copyBuffer(oldBuffer,newBuffer,oldOffset,newOffset,copyRangeLen);
}

size_t          GPUMemoryAllocatorBase::min_alignment() const noexcept
{
    return mDriver->getMinimumMemoryMapAlignment();
}

IVideoDriver*   GPUMemoryAllocatorBase::getDriver() noexcept
{
    return mDriver;
}

