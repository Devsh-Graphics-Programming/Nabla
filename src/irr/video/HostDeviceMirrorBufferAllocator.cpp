#include "irr/video/HostDeviceMirrorBufferAllocator.h"
#include "IVideoDriver.h"

using namespace irr;
using namespace video;

decltype(HostDeviceMirrorBufferAllocator::lastAllocation) HostDeviceMirrorBufferAllocator::createBuffers(size_t bytes, size_t alignment)
{
    decltype(HostDeviceMirrorBufferAllocator::lastAllocation) retval;
    retval.first = _IRR_ALIGNED_MALLOC(bytes,alignment);

    auto memReqs = mDriver->getDeviceLocalGPUMemoryReqs();
    memReqs.vulkanReqs.size = bytes;
    retval.second = mDriver->createGPUBufferOnDedMem(memReqs,false);

    return retval;
}
