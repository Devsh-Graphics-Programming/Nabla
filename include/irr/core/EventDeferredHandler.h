#ifndef __IRR_EVENT_DEFERRED_HANDLER_H__
#define __IRR_EVENT_DEFERRED_HANDLER_H__


#include "irr/core/Types.h"

namespace irr
{
namespace core
{

template<class Event, class Functor>
class EventDeferredHandlerST
{
    public:
        typedef std::pair<Event,Functor>                  DeferredEvent;
    protected:
        typedef core::forward_list<DeferredEvent> EventContainerType;
		uint32_t								mEventsCount;
        EventContainerType                      mEvents;
        typename EventContainerType::iterator	mLastEvent;
    public:
        EventDeferredHandlerST() : mEventsCount(0u)
        {
            mLastEvent = mEvents.before_begin();
        }

        virtual ~EventDeferredHandlerST()
        {
            while (mEventsCount)
            {
                auto prev = mEvents.before_begin();
                for (auto it = mEvents.begin(); it!=mEvents.end(); )
                {
                    if (it->first.wait_until(std::chrono::high_resolution_clock::now()+std::chrono::microseconds(250ull)))
                    {
                        it->second();
                        it = mEvents.erase_after(prev);
                        mEventsCount--;
                        continue;
                    }
                    prev = it++;
                }
                mLastEvent = prev;
            }
        }

        inline void     addEvent(Event&& event, Functor&& functor)
        {
            mLastEvent = mEvents.emplace_after(mLastEvent,std::forward<Event>(event),std::forward<Functor>(functor));
            mEventsCount++;
        }
        /*
        //! Abort does not call the operator()
        inline uint32_t abortEvent(const Event& eventToAbort)
        {
            #ifdef _IRR_DEBUG
            assert(mEvents.size());
            #endif // _IRR_DEBUG
            std::remove_if(mEvents.begin(),mEvents.end(),[&](const DeferredEvent& x){return x.first==eventToAbort;});
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
                        it->second->grab();
                        auto oldFence = it2->first;
                        it2->first = it->second;
                        oldFence->drop();
                    }
                }
            }
        }
        */


        template<class Clock, class Duration, typename... Args>
        inline uint32_t waitUntilForReadyEvents(const std::chrono::time_point<Clock, Duration>& timeout_time, Args&... args)
        {
            while (mEventsCount)
            {
                // iterate to poll
                auto prev = mEvents.before_begin();
                for (auto it = mEvents.begin(); it!=mEvents.end();)
                {
                    bool success;
                    auto currentTime = Clock::now();
                    bool canWait = timeout_time>currentTime;
                    if (canWait)
                    {
                        std::chrono::time_point<Clock> singleWaitTimePt((currentTime.time_since_epoch()*(mEventsCount-1u)+timeout_time.time_since_epoch())/mEventsCount);
                        success = it->first.wait_until(singleWaitTimePt);
                    }
                    else
                        success = it->first.poll();

                    if (success)
                    {
                        bool earlyQuit = it->second(args...);
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
                if (it->first.poll())
                {
                    bool earlyQuit = it->second(args...);
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
                if (it->first.poll())
                {
                    it->second();
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




