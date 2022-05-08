#ifndef __NBL_I_GPU_FENCE_H_INCLUDED__
#define __NBL_I_GPU_FENCE_H_INCLUDED__


#include "nbl/core/declarations.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class NBL_API IGPUFence : public core::IReferenceCounted, public IBackendObject
{
    public:
        enum E_CREATE_FLAGS : uint32_t
        {
            ECF_UNSIGNALED = 0x00u,
            ECF_SIGNALED_BIT = 0x01u
        };
        enum E_STATUS
        {
            ES_SUCCESS,
            ES_TIMEOUT,
            ES_NOT_READY,
            ES_ERROR
        };

        IGPUFence(core::smart_refctd_ptr<const ILogicalDevice>&& dev, E_CREATE_FLAGS flags) : IBackendObject(std::move(dev))
        {
        }

        // OpenGL: core::smart_refctd_ptr<COpenGLSync>*
        // Vulkan: const VkFence*
        virtual void* getNativeHandle() = 0;

    protected:
        virtual ~IGPUFence() = default;
};


class NBL_API GPUEventWrapper : public core::Uncopyable
{
protected:
    ILogicalDevice* mDevice;
    core::smart_refctd_ptr<IGPUFence> mFence;
public:
    GPUEventWrapper(ILogicalDevice* dev, core::smart_refctd_ptr<IGPUFence>&& fence) : mDevice(dev), mFence(std::move(fence))
    {
    }
    GPUEventWrapper(const GPUEventWrapper& other) = delete;
    GPUEventWrapper(GPUEventWrapper&& other) noexcept : mFence(nullptr)
    {
        this->operator=(std::forward<GPUEventWrapper>(other));
    }
    virtual ~GPUEventWrapper()
    {
    }

    GPUEventWrapper& operator=(const GPUEventWrapper& other) = delete;
    inline GPUEventWrapper& operator=(GPUEventWrapper&& other) noexcept
    {
        mFence.operator=(std::move(other.mFence));
        mDevice = other.mDevice;
        return *this;
    }

    template<class Clock=std::chrono::steady_clock, class Duration=typename Clock::duration>
    inline static std::chrono::time_point<Clock,Duration> default_wait()
    {
        //return typename Clock::now()+std::chrono::nanoseconds(50000ull); // 50 us
        return Clock::now()+std::chrono::nanoseconds(50000ull); // 50 us
    }

    IGPUFence::E_STATUS waitFenceWrapper(IGPUFence* fence, uint64_t timeout);
    IGPUFence::E_STATUS getFenceStatusWrapper(IGPUFence* fence);

    template<class Clock=std::chrono::steady_clock, class Duration=typename Clock::duration>
    inline bool wait_until(const std::chrono::time_point<Clock,Duration>& timeout_time)
    {
        auto currentClockTime = Clock::now();
        do
        {
            uint64_t nanosecondsLeft = 0ull;
            if (currentClockTime<timeout_time)
                nanosecondsLeft = std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_time-currentClockTime).count();
            const IGPUFence::E_STATUS waitStatus = waitFenceWrapper(mFence.get(), nanosecondsLeft);
            switch (waitStatus)
            {
            case IGPUFence::ES_ERROR:
                return true;
            case IGPUFence::ES_TIMEOUT:
                break;
            case IGPUFence::ES_SUCCESS:
                return true;
                break;
            }
            currentClockTime = Clock::now();
        } while (currentClockTime<timeout_time);

        return false;
    }

    inline bool poll()
    {
        const IGPUFence::E_STATUS status = getFenceStatusWrapper(mFence.get());
        switch (status)
        {
        case IGPUFence::ES_ERROR:
        case IGPUFence::ES_SUCCESS:
            return true;
            break;
        default:
            break;
        }
        return false;
    }

    inline bool operator==(const GPUEventWrapper& other)
    {
        return mFence==other.mFence;
    }
    inline bool operator<(const GPUEventWrapper& other)
    {
        return mFence.get()<other.mFence.get();
    }
};

template<class Functor>
using GPUDeferredEventHandlerST = core::DeferredEventHandlerST<core::DeferredEvent<GPUEventWrapper,Functor> >;

}

#endif