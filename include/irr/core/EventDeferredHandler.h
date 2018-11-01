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
        typedef std::pair<Event,Functor>    DeferredEvent;
    protected:
        core::vector<DeferredEvent>         mEvents;
    public:
        EventDeferredHandlerST() {}

        virtual ~EventDeferredHandlerST()
        {
            while (mEvents.size())
            {
                pollForReadyEvents();
            }
        }

        inline void     addEvent(Event&& event, Functor&& functor)
        {
            mEvents.emplace_back(std::forward<Event>(event),std::forward<Functor>(functor));
        }
        //! Abort does not call the operator()
        inline uint32_t abortEvent(const Event& eventToAbort)
        {
            #ifdef _DEBUG
            assert(mEvents.size());
            #endif // _DEBUG
            std::remove_if(mEvents.begin(),mEvents.end(),[&](const DeferredEvent& x){return x.first==eventToAbort;});
            return mEvents.size();
        }
        /** For later implementation -- WARNING really old code
        inline void     swapEvents()
        {
            // extras from old StreamingTransientDataBuffer
            inline void         swap_fences(std::pair<const IDriverFence*,IDriverFence*>* swaplist_begin,
                                            std::pair<const IDriverFence*,IDriverFence*>* swaplist_end) noexcept
            {
                std::sort(deferredFrees.begin(),deferredFrees.end(),deferredFreeCompareFunc);

                for (auto it=swaplist_begin; it!=swaplist_end; it++)
                {
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

        template<class Clock, class Duration>
        inline uint32_t waitUntilForReadyEvents(const std::chrono::time_point<Clock, Duration>& timeout_time)
        {
            return waitUntilForReadyEvents(timeout_time,~size_t(0u));
        }

        inline uint32_t pollForReadyEvents()
        {
            return pollForReadyEvents(~size_t(0u));
        }


        template<class Clock, class Duration>
        inline uint32_t waitUntilForReadyEvents(const std::chrono::time_point<Clock, Duration>& timeout_time, size_t maxEventsToPoll)
        {
            while (mEvents.size())
            {
                // iterate to poll
                for (auto it = mEvents.begin(); it!=mEvents.end();)
                {
                    if (maxEventsToPoll--)
                        return mEvents.size();

                    bool success;
                    auto currentTime = Clock::now();
                    bool canWait = timeout_time>currentTime;
                    if (canWait)
                    {
                        Duration singleWaitTimePt = (currentTime*(mEvents.size()-1u)+timeout_time)/mEvents.size();
                        success = it->first.wait_until(singleWaitTimePt);
                    }
                    else
                        success = it->first.poll();

                    if (success)
                    {
                        it->second();
                        it = mEvents.erase(it);
                        continue;
                    }
                    // dont care about timeout until we hit first fence we had to wait for
                    if (!canWait)
                        return mEvents.size();

                    it++;
                }
            }

            return 0u;
        }

        inline uint32_t pollForReadyEvents(size_t maxEventsToPoll)
        {
            for (auto it = mEvents.begin(); (maxEventsToPoll--) && it!=mEvents.end();)
            {
                if (it->first.poll())
                {
                    it->second();
                    it = mEvents.erase(it);
                    continue;
                }

                it++;
            }

            return mEvents.size();
        }
};

//! EventDeferredHandlerMT coming later

}
}

#endif // __IRR_EVENT_DEFERRED_HANDLER_H__




