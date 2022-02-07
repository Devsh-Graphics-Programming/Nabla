// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_I_DRIVER_FENCE_H_INCLUDED__
#define __NBL_I_DRIVER_FENCE_H_INCLUDED__

#include <chrono>
#include "nbl/core/BaseClasses.h"
#include "nbl/core/EventDeferredHandler.h"

namespace nbl
{
namespace video
{
enum E_DRIVER_FENCE_RETVAL
{
    //! Indicates that an error occurred. Additionally, an OpenGL error will be generated.
    EDFR_FAIL = 0,
    //! If it returns GL_TIMEOUT_EXPIRED, then the sync object did not signal within the given timeout period (includes before us calling the func).
    EDFR_TIMEOUT_EXPIRED,
    //! Indicates that sync​ was signaled before the timeout expired.
    EDFR_CONDITION_SATISFIED,
    //! GPU already completed work before we even asked == THIS IS WHAT WE WANT
    EDFR_ALREADY_SIGNALED
};

//! Base class for Fences, persistently Mapped buffer
/*
    Using fences is neccesarry while requesting operation for
    GPU depended on CPU side, like fetching mapped GPU buffer.
    It's significant because CPU and GPU execute processes asynchronously,
    so before the CPU does anything for instance to mapped memory the GPU
    has been writing to, waiting on a fence is needed and then if the mapped memory
    needs to be invalidated before CPU reads it.

    So basically you should use \bcanDeferredFlush()\b.
    It is a property of \bIDriverFence\b and it tells you whether 
    the \bwaitCPU()\b method can accept a 
    
    \code{.cpp}
    flush = true
    \endcode
  
    parameter (which performs an implicit \bglFlush\b on the context you've placed 
    the fence on just before you actually start witing on the fence)

    @see waitCPU

    A pipeline/context flush (\bglFlush\b) flushes the driver-CPU-side queue onto the 
    GPU device for execution. If you place a fence but don't flush you can end up in a 
    deadlock. Generally each time you call \bwaitCPU()\b it will time-out because the work 
    has not been sent to the GPU for execution 100% (it's stuck in transit).

    So you either need to perform a \bglFlush\b (but on the same thread and context that 
    placed the fence) after placing the fence, but before \bwaitCPU()\b execution (on any 
    thread) or use that implicit flush functionality (but only if setting thread is the 
    waiting thread).

    One of example of fence usage is as follows when writing to GPU buffer
    with screen shot extension:

    \code{.cpp}
    auto fence = ext::ScreenShot::createScreenShot(driver, gpuimg, buffer.get());
    while (fence->waitCPU(1000ull, fence->canDeferredFlush()) == video::EDFR_TIMEOUT_EXPIRED)
    {
        // do something while waiting on GPU
    }
    \endcode

    Also you should look at IDriver::flushMappedMemoryRanges function. 
    It flushes when CPU is writing and GPU is reading.

    @see IDriver::flushMappedMemoryRanges

    Notes:

    - if you're not confident controlling cache coherency manually then use the \bCOHERENT\b 
    memory type when creating mappable IGPUBuffers

    @see IGPUBuffer
*/

class IDriverFence : public core::IReferenceCounted
{
    _NBL_INTERFACE_CHILD(IDriverFence) {}

public:
    //! This tells us if we can set the `flush` argument of the `waitCPU` function to true
    virtual bool canDeferredFlush() const = 0;

    //! If timeout​ is zero, the function will simply check to see if the sync object is signaled and return immediately.
    /** \param timeout in nanoseconds.
        \param whether to perform a special implicit flush in OpenGL (quite useless).
        IMPORTANT: In OpenGL you need to glFlush at some point in the thread (context) AFTER you create this fence with
        IVideoDriver::placeFence and BEFORE you waitCPU on it in the same thread or another.
        https://www.khronos.org/opengl/wiki/Sync_Object#Flushing_and_contexts
        If you don't THE FENCE MAY NEVER BE SIGNALLED because the signalling command has never been flushed to the device queue.
        The `flush` parameter can do this glFlush for you just before the wait, BUT ONLY IF you've placed the fence in the same
        thread as the one you are waiting on the fence. So if you want to use IDriverFence for inter-context coordination you
        are screwed and must call glFlush manually.*/
    virtual E_DRIVER_FENCE_RETVAL waitCPU(const uint64_t& timeout, const bool& flush = false) = 0;

    //! This makes the GPU pause executing commands in the current context until commands before the fence in the context which created it, have completed
    /** You may be shocked to learn that OpenGL allows for commands in the same context to execute simultaneously or out of order as long as the result
        of these commands is the same as if they have been executed strictly in-order (except the memory effects on the objects following the incoherent memory model).
        For solving the above within a context you want to issue memory barriers, however for ensuring the order of commands between contexts you want to use waitGPU.*/
    virtual void waitGPU() = 0;
};

class GPUEventWrapper : public core::Uncopyable
{
protected:
    core::smart_refctd_ptr<IDriverFence> mFence;

public:
    GPUEventWrapper(core::smart_refctd_ptr<IDriverFence>&& fence)
        : mFence(std::move(fence))
    {
    }
    GPUEventWrapper(const GPUEventWrapper& other) = delete;
    GPUEventWrapper(GPUEventWrapper&& other) noexcept
        : mFence(nullptr)
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
        return *this;
    }

    template<class Clock = std::chrono::steady_clock, class Duration = typename Clock::duration>
    inline static std::chrono::time_point<Clock, Duration> default_wait()
    {
        return std::chrono::high_resolution_clock::now() + std::chrono::nanoseconds(50000ull);  // 50 us
    }
    template<class Clock = std::chrono::steady_clock, class Duration = typename Clock::duration>
    inline bool wait_until(const std::chrono::time_point<Clock, Duration>& timeout_time)
    {
        auto currentClockTime = Clock::now();
        do
        {
            uint64_t nanosecondsLeft = 0ull;
            if(currentClockTime < timeout_time)
                nanosecondsLeft = std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_time - currentClockTime).count();
            switch(mFence->waitCPU(nanosecondsLeft, mFence->canDeferredFlush()))
            {
                case EDFR_FAIL:
                    return true;
                case EDFR_TIMEOUT_EXPIRED:
                    break;
                case EDFR_CONDITION_SATISFIED:
                case EDFR_ALREADY_SIGNALED:
                    return true;
                    break;
            }
            currentClockTime = Clock::now();
        }
        while(currentClockTime < timeout_time);

        return false;
    }

    inline bool poll()
    {
        switch(mFence->waitCPU(0u, mFence->canDeferredFlush()))
        {
            case EDFR_FAIL:
            case EDFR_CONDITION_SATISFIED:
            case EDFR_ALREADY_SIGNALED:
                return true;
                break;
            default:
                break;
        }
        return false;
    }

    inline bool operator==(const GPUEventWrapper& other)
    {
        return mFence == other.mFence;
    }
    inline bool operator<(const GPUEventWrapper& other)
    {
        return mFence.get() < other.mFence.get();
    }
};

template<class Functor>
using GPUDeferredEventHandlerST = core::DeferredEventHandlerST<core::DeferredEvent<GPUEventWrapper, Functor> >;

}  // end namespace scene
}  // end namespace nbl

#endif
