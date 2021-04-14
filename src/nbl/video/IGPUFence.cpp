#include "nbl/video/IGPUFence.h"

#include "nbl/video/ILogicalDevice.h"

namespace nbl {
namespace video
{

IGPUFence::E_STATUS GPUEventWrapper::waitFenceWrapper(IGPUFence* fence, uint64_t timeout)
{
    return mDevice->waitForFences(1u, &fence, false, timeout);
}

IGPUFence::E_STATUS GPUEventWrapper::getFenceStatusWrapper(IGPUFence* fence)
{
    return mDevice->getFenceStatus(fence);
}

}
}