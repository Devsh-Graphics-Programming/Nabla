#ifndef _NBL_VIDEO_I_SEMAPHORE_H_INCLUDED_
#define _NBL_VIDEO_I_SEMAPHORE_H_INCLUDED_


#include "nbl/core/IReferenceCounted.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class ISemaphore : public core::IReferenceCounted, public IBackendObject
{
    public:
        constexpr static inline uint64_t InvalidCounterValue = ~0ull;
        inline uint64_t getCounterValue() const
        {
            if (m_isTimeline)
                return getCounterValue_impl();
            assert(false);
            return InvalidCounterValue;
        }

        //! Basically the counter can only monotonically increase with time (ergo the "timeline"):
        // 1. `value` must have a value greater than the current value of the semaphore (what you'd get from `getCounterValue()`)
        // 2. `value` must be less than the value of any pending semaphore signal operations (this is actually more complicated)
        // Current pending signal operations can complete in any order, unless there's an execution dependency between them,
        // this will change the current value of the semaphore. Consider a semaphore with current value of 2 and pending signals of 3,4,5;
        // without any execution dependencies, you can only signal a value higher than 2 but less than 3 which is impossible.
        inline void signal(const uint64_t value)
        {
            assert(m_isTimeline);
            if (m_isTimeline)
                signal_impl(value);
        }

        // Vulkan: const VkSemaphore*
        virtual const void* getNativeHandle() const = 0;

    protected:
        inline ISemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const bool _isTimeline) : IBackendObject(std::move(dev)), m_isTimeline(_isTimeline) {}
        virtual ~ISemaphore() = default;

        virtual uint64_t getCounterValue_impl() const = 0;
        virtual void signal_impl(const uint64_t value) = 0;


        const bool m_isTimeline;
};

}

#endif