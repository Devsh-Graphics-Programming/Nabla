#include "nbl/video/ISemaphore.h"
#include "nbl/video/ILogicalDevice.h"

namespace nbl::video
{

bool TimelineEventHandlerBase::notTimedOut(const uint64_t value, const uint64_t nanoseconds)
{
    const ILogicalDevice::SSemaphoreWaitInfo info = {.semaphore=m_sema.get(),.value=value};
    switch (const_cast<ILogicalDevice*>(m_sema->getOriginDevice())->waitForSemaphores({&info,1},true,nanoseconds))
    {
        case ILogicalDevice::WAIT_RESULT::TIMEOUT:
            return false;
            break;
        default: 
            break;
    }
    return true;
}

}