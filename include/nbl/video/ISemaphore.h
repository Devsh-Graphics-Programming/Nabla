#ifndef _NBL_VIDEO_I_SEMAPHORE_H_INCLUDED_
#define _NBL_VIDEO_I_SEMAPHORE_H_INCLUDED_


#include "nbl/core/IReferenceCounted.h"

#include <chrono>

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class ISemaphore : public IBackendObject
{
    public:
        // basically a pool function
        virtual uint64_t getCounterValue() const = 0;

        //! Basically the counter can only monotonically increase with time (ergo the "timeline"):
        // 1. `value` must have a value greater than the current value of the semaphore (what you'd get from `getCounterValue()`)
        // 2. `value` must be less than the value of any pending semaphore signal operations (this is actually more complicated)
        // Current pending signal operations can complete in any order, unless there's an execution dependency between them,
        // this will change the current value of the semaphore. Consider a semaphore with current value of 2 and pending signals of 3,4,5;
        // without any execution dependencies, you can only signal a value higher than 2 but less than 3 which is impossible.
        virtual void signal(const uint64_t value) = 0;

        // We don't provide waits as part of the semaphore (cause you can await multiple at once with ILogicalDevice),
        // but don't want to pollute ILogicalDevice with lots of enums and structs
        struct SWaitInfo
        {
            const ISemaphore* semaphore;
            uint64_t value;
        };
        enum class WAIT_RESULT : uint8_t
        {
            TIMEOUT,
            SUCCESS,
            DEVICE_LOST,
            _ERROR
        };

        // Vulkan: const VkSemaphore*
        virtual const void* getNativeHandle() const = 0;

    protected:
        inline ISemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& dev) : IBackendObject(std::move(dev)) {}
        virtual ~ISemaphore() = default;
};

}
#endif