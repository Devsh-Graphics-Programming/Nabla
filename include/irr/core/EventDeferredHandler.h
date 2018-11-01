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
        static inline bool  eventCompareFunction(const DeferredEvent& comp, const Event& val) {return comp.first<val;}

        core::vector<DeferredEvent>         mEvents;
        bool                                mEventsNeedSort;
        inline void         sortEvents()
        {
            if (!mEventsNeedSort)
                return;

            std::sort(mEvents.begin(),mEvents.end(),eventCompareFunction);
            mEventsNeedSort = false;
        }
    public:
        EventDeferredHandlerST() : mEvents(), mEventsNeedSort(false) {}

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
        inline uint32_t abortEvent(const Event& eventToAbort)
        {
            #ifdef _DEBUG
            assert(mEvents.size());
            #endif // _DEBUG
            sortEvents();

            auto deletionStart = std::lower_bound(mEvents.begin(),mEvents.end(),eventToAbort,eventCompareFunction);
            if (deletionStart==mEvents.end())
                return 0u;

            auto deletetionEnd = std::upper_bound(deletionStart,mEvents.end(),eventToAbort,eventCompareFunction);
            auto deletedCount = std::distance(deletionStart,deletetionEnd);

            mEvents.erase(deletionStart,deletetionEnd);

            return deletedCount;
        }
        /** For later implementation
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
            if (mEvents.size()<2u)
            {
                if (mEvents.size() && mEvents[0].wait_until(timeout_time))
                    mEvents.clear();

                return mEvents.size();
            }

            sortEvents();

            while (mEvents.size())
            {
                // iterate to poll
                for (auto it = mEvents.begin(); it!=mEvents.end();)
                {
                    if (maxEventsToPoll--)
                        return mEvents.size();

                    auto next = std::upper_bound(it,mEvents.end(),eventCompareFunction);
                    if (it->poll())
                    {
                        it = mEvents.erase(it,next);
                        continue;
                    }
                    // dont care about timeout until we hit first fence we have to wait for
                    if (Clock::now()>timeout_time)
                        return mEvents.size();

                    it = next;
                }
            }

            return 0u;
        }

        inline uint32_t pollForReadyEvents(size_t maxEventsToPoll)
        {
            sortEvents();

            // iterate through unique fences
            for (auto it = mEvents.begin(); (maxEventsToPoll--) && it!=mEvents.end();)
            {
                auto next = std::upper_bound(it,mEvents.end(),eventCompareFunction);
                if (it->poll())
                {
                    it = mEvents.erase(it,next);
                    continue;
                }

                it = next;
            }

            return mEvents.size();
        }
};

//! EventDeferredHandlerMT coming later

}
}

#endif // __IRR_EVENT_DEFERRED_HANDLER_H__




