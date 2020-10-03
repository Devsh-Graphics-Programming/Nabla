// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __IRR_EVENT_DEFERRED_HANDLER_H__
#define __IRR_EVENT_DEFERRED_HANDLER_H__


#include "irr/core/Types.h"

namespace irr
{
namespace core
{

template<class Event, class Functor>
struct DeferredEvent
{
    using event_t = Event;
    using functor_t = Functor;

    DeferredEvent(event_t&& _event, functor_t&& _function) : m_event(std::move(_event)), m_function(std::move(_function)) {}
    DeferredEvent(DeferredEvent&& other)
    {
        operator=(std::move(other));
    }

    inline DeferredEvent& operator=(DeferredEvent&& other)
    {
        std::swap(m_event, other.m_event);
        std::swap(m_function, other.m_function);
        return *this;
    }

    event_t m_event;
    functor_t m_function;
};

class PolymorphicEvent : public core::Uncopyable
{
    protected:
        PolymorphicEvent() = default;
        virtual ~PolymorphicEvent() {}

        enum WaitRes
        {
            Fail,
            Timeout,
            Done
        };
        virtual WaitRes wait_for(const std::chrono::nanoseconds& len) = 0;

    public:
        PolymorphicEvent& operator=(const PolymorphicEvent& other) = delete;
        virtual PolymorphicEvent& operator=(PolymorphicEvent&& other) noexcept
        {
        }

        template<class Clock, class Duration>
        inline bool wait_until(const std::chrono::time_point<Clock, Duration>& timeout_time)
        {
            auto currentClockTime = Clock::now();
            do
            {
                uint64_t nanosecondsLeft = 0ull;
                if (currentClockTime<timeout_time)
                    nanosecondsLeft = std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_time-currentClockTime).count();
                switch (wait_for(nanosecondsLeft))
                {
                    case Fail:
                        return true;
                        break;
                    case Done:
                        return true;
                        break;
                    default: // Timeout
                        break;
                }
                currentClockTime = Clock::now();
            } while (currentClockTime<timeout_time);

            return false;
        }

        virtual bool poll() = 0;

        virtual bool operator==(const PolymorphicEvent& other)
        {
            return false;
        }
};

template<class Functor>
using DeferredPolymorphicEvent = DeferredEvent<PolymorphicEvent*,Functor>;

template<class Event>
using DeferredEventPolymorphic = DeferredEvent<Event,std::function<void()> >;

using DeferredPolymorphicEventPolymorphic = DeferredEvent<PolymorphicEvent*,std::function<void()> >;


template<class DeferredEvent>
class DeferredEventHandlerST
{
    protected:
        using EventContainerType = core::forward_list<DeferredEvent>;
		uint32_t								mEventsCount;
        EventContainerType                      mEvents;
        typename EventContainerType::iterator	mLastEvent;

    public:
        using event_t = typename DeferredEvent::event_t;
        using functor_t = typename DeferredEvent::functor_t;

        DeferredEventHandlerST() : mEventsCount(0u)
        {
            mLastEvent = mEvents.before_begin();
        }

        virtual ~DeferredEventHandlerST()
        {
            while (mEventsCount)
            {
                auto prev = mEvents.before_begin();
                for (auto it = mEvents.begin(); it!=mEvents.end(); )
                {
                    if (it->m_event.wait_until(std::chrono::high_resolution_clock::now()+std::chrono::microseconds(250ull)))
                    {
                        it->m_function();
                        it = mEvents.erase_after(prev);
                        mEventsCount--;
                        continue;
                    }
                    prev = it++;
                }
                mLastEvent = prev;
            }
        }

        inline void     addEvent(event_t&& event, functor_t&& functor)
        {
            mLastEvent = mEvents.emplace_after(mLastEvent,std::forward<event_t>(event),std::forward<functor_t>(functor));
            mEventsCount++;
        }
        /*
        //! Abort does not call the operator()
        inline uint32_t abortEvent(const event_t& eventToAbort)
        {
            #ifdef _IRR_DEBUG
            assert(mEvents.size());
            #endif // _IRR_DEBUG
            std::remove_if(mEvents.begin(),mEvents.end(),[&](const DeferredEvent& x){return x.m_event==eventToAbort;});
            mLastEvent = ?
            return mEvents.size();
        }
        // For later implementation -- WARNING really old code
        inline void     swapEvents()
        {
            // extras from old StreamingTransientDataBuffer
            inline void         swap_fences(std::pair<const IDriverFence*,IDriverFence*>* swaplist_begin,
                                            std::pair<const IDriverFence*,IDriverFence*>* swaplist_end) noexcept
            {
                std::sort(deferredFrees.begin(),deferredFrees.end(),deferredFreeCompareFunc);

                for (auto it=swaplist_begin; it!=swaplist_end; it++)
                {
                    //! WARNING The deferredFrees has changed into a forward list!
                    auto lbound = std::lower_bound(deferredFrees.begin(),deferredFrees.end(),*it,deferredFreeCompareFunc);
                    auto ubound = std::upper_bound(lbound,deferredFrees.end(),*it,deferredFreeCompareFunc);
                    for (auto it2=lbound; it2!=ubound; it2++)
                    {
                        it->m_function->grab();
                        auto oldFence = it2->m_event;
                        it2->m_event = it->m_function;
                        oldFence->drop();
                    }
                }
            }
        }
        */


        template<class Clock, class Duration, typename... Args>
        inline uint32_t waitUntilForReadyEvents(const std::chrono::time_point<Clock, Duration>& timeout_time, Args&... args)
        {
            // keep on iterating until there are no events left, we time out or functor tells us we can quit early
            while (mEventsCount)
            {
                // iterate to poll, from oldest to newest (oldest event most likely to signal first)
                auto prev = mEvents.before_begin();
                for (auto it = mEvents.begin(); it!=mEvents.end();)
                {
                    bool success;
                    auto currentTime = Clock::now();
                    bool canWait = timeout_time>currentTime;
                    if (canWait)
                    {
                        // want to give each event equal wait time, so interpolate (albeit weirdly)
                        std::chrono::time_point<Clock> singleWaitTimePt((currentTime.time_since_epoch()*(mEventsCount-1u)+timeout_time.time_since_epoch())/mEventsCount);
                        success = it->m_event.wait_until(singleWaitTimePt);
                    }
                    else
                        success = it->m_event.poll();

                    if (success)
                    {
                        bool earlyQuit = it->m_function(args...);
                        it = mEvents.erase_after(prev);
                        mEventsCount--;
                        if (earlyQuit)
                        {
                            if (it==mEvents.end())
                                mLastEvent = prev;
                            return mEventsCount;
                        }

                        continue;
                    }
                    // dont care about timeout until we hit first fence we had to wait for
                    if (!canWait)
                        return mEventsCount;

                    prev = it++;
                }
                mLastEvent = prev;
            }

            return 0u;
        }

        template<typename... Args>
        inline uint32_t pollForReadyEvents(Args&... args)
        {
            auto prev = mEvents.before_begin();
            for (auto it = mEvents.begin(); it!=mEvents.end();)
            {
                if (it->m_event.poll())
                {
                    bool earlyQuit = it->m_function(args...);
                    it = mEvents.erase_after(prev);
                    mEventsCount--;
                    if (earlyQuit)
                    {
                        if (it==mEvents.end())
                            mLastEvent = prev;
                        return mEventsCount;
                    }

                    continue;
                }
                prev = it++;
            }
            mLastEvent = prev;

            return mEventsCount;
        }

        //! Will try to poll enough events so that the number of events in the queue is less or equal to maxEventCount
        template<typename... Args>
        inline uint32_t cullEvents(uint32_t maxEventCount)
        {
            if (mEventsCount<=maxEventCount)
                return mEventsCount;

            auto prev = mEvents.before_begin();
            for (auto it = mEvents.begin(); mEventsCount>maxEventCount&&it!=mEvents.end();)
            {
                if (it->m_event.poll())
                {
                    it->m_function();
                    it = mEvents.erase_after(prev);
                    mEventsCount--;
                    continue;
                }
                prev = it++;
            }
            mLastEvent = prev;

            return mEventsCount;
        }
};

//! EventDeferredHandlerMT coming later

}
}

#endif // __IRR_EVENT_DEFERRED_HANDLER_H__