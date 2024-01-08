#ifndef _NBL_VIDEO_TIMELINE_EVENT_HANDLERS_H_INCLUDED_
#define _NBL_VIDEO_TIMELINE_EVENT_HANDLERS_H_INCLUDED_


#include "nbl/video/ILogicalDevice.h"

#include <chrono>


namespace nbl::video
{

// Could be made MT and relatively lockless, if only had a good lock-few circular buffer impl
// Not sure its worth the effort as anything using this will probably need to be lockful to be MT
template<typename Functor>
class TimelineEventHandlerST final : core::Unmovable, core::Uncopyable
{
        struct FunctorValuePair
        {
            Functor func;
            uint64_t geSemaValue;
        };
        // could be a circular buffer but whatever for now
        core::deque<FunctorValuePair> m_cb;
        core::smart_refctd_ptr<ISemaphore> m_sema;
        uint64_t m_greatestSignal;
        uint64_t m_greatestLatch;
        
        template<bool QueryCounter=true, typename Lambda>
        inline uint32_t for_each_popping(Lambda&& l)
        {
            if (m_cb.empty())
                return 0;

            if (QueryCounter)
                m_greatestSignal = m_sema->getCounterValue();
            // In a threadsafe scenario, you'd immediately pop everything you can with geSemaValue<=signal
            // the way that it would happen is we'd `reserveLock` everything in the buffer so far
            // then rewind the reservation for anything that doesn't meet the predicate.
            // For this to work, the predicate needs to be "consistent" meaning no holes can be formed by multiple actors.
            while (!m_cb.empty() && l(m_cb.front()))
                m_cb.pop_front();
            m_greatestLatch = m_cb.empty() ? 0:m_cb.back().geSemaValue;
            return static_cast<uint32_t>(m_cb.size());
        }

        inline auto constructNonBailing()
        {
            return [&](FunctorValuePair& p) -> bool
            {
                if (p.geSemaValue>m_greatestSignal)
                    return false;
                p.func();
                return true;
            };
        }
        inline auto constructBailing(bool& bailed)
        {
            return [&](FunctorValuePair& p) -> bool
            {
                if (p.geSemaValue>m_greatestSignal)
                    return false;
                const bool last_bailed = bailed;
                bailed = p.func();
                return !last_bailed;
            };
        }

        // If the functor returns bool, then we bail on the on the first executed event during wait,poll,etc.
        constexpr static inline bool ReturnsBool = std::is_same_v<decltype(std::declval<Functor>()()),bool>;

    public:
        // Theoretically could make a factory function cause passing a null semaphore is invalid, but counting on users to be relatively intelligent.
        inline TimelineEventHandlerST(core::smart_refctd_ptr<ISemaphore>&& sema, const uint64_t initialCapacity = 4095 / sizeof(FunctorValuePair) + 1) :
            m_sema(std::move(sema)), m_greatestSignal(m_sema->getCounterValue()), m_greatestLatch(0) {}
        // If you don't want to deadlock here, look into the `abort*` family of methods
        ~TimelineEventHandlerST()
        {
            while (wait(std::chrono::steady_clock::now()+std::chrono::seconds(5))) {}
        }
        // little utility
        inline ISemaphore* getSemaphore() const {return m_sema.get();}

        inline uint32_t count() const {return m_cb.size();}

        // You can latch arbitrary functors upon the semaphore reaching some value
        inline void latch(const uint64_t geSemaValue, Functor&& function)
        {
            //const auto oldValue = core::atomic_fetch_max(&m_greatestLatch,geSemaValue);
            assert(geSemaValue>=m_greatestLatch); // you cannot latch out of order
            m_greatestLatch = geSemaValue;
            m_cb.emplace_back(std::move(function),geSemaValue);
        }

        // Returns number of events still outstanding
        inline uint32_t poll(bool& bailed)
        {
            bailed = false;
            if constexpr (ReturnsBool)
                return for_each_popping(constructBailing(bailed));
            else
                return for_each_popping(constructNonBailing());
        }
        inline uint32_t poll()
        {
            bool dummy;
            return poll(dummy);
        }

        template<class Clock, class Duration=typename Clock::duration>
        inline uint32_t wait(const std::chrono::time_point<Clock,Duration>& timeout_time)
        {
            if (m_cb.empty())
                return 0;

            auto singleSemaphoreWait = [&](const uint64_t waitVal, const std::chrono::time_point<Clock,Duration>& waitPoint)->uint64_t
            {
                const auto current_time = Clock::now();
                if (waitPoint>current_time)
                {
                    auto device = const_cast<ILogicalDevice*>(m_sema->getOriginDevice());
                    const auto nanosecondsLeft = std::chrono::duration_cast<std::chrono::nanoseconds>(waitPoint-current_time).count();
                    const ISemaphore::SWaitInfo info = {.semaphore=m_sema.get(),.value = waitVal};
                    if (device->waitForSemaphores({&info,1},true,nanosecondsLeft)==ISemaphore::WAIT_RESULT::SUCCESS)
                        return waitVal>m_greatestSignal ? waitVal:m_greatestSignal; // remeber that latch can move back, not signal though
                }
                return m_sema->getCounterValue();
            };

            if constexpr (ReturnsBool)
            {
                // Perf-assumption: there are probably no latched events with wait values less or equal to `m_greatestSignal`
                // So we have a bunch of events with semaphore values between `m_greatestSignal` and `m_greatestLatch` with
                // lots of repeated latch values incrementing by a fixed K amount between each batch of repeats
                auto currentTime = Clock::now();
                do
                {
                    // We cannot wait for the original timeout point because we want to be able to bail, so increment slowly
                    const auto uniqueValueEstimate = core::min(m_cb.size(),m_greatestSignal-m_greatestLatch);
                    // weird interpolation that works on integers, basically trying to get somethign 1/uniqueValueEstimate of the way from now to original timeout point
                    const std::chrono::time_point<Clock> singleWaitTimePt((currentTime.time_since_epoch()*(uniqueValueEstimate-1u)+timeout_time.time_since_epoch())/uniqueValueEstimate);
                    // So we only Semaphore wait for the next latch value we need
                    m_greatestSignal = singleSemaphoreWait(m_cb.front().geSemaValue,singleWaitTimePt);

                    bool bailed = false;
                    for_each_popping<false>(constructBailing(bailed));
                    if (bailed)
                        break;
                } while ((currentTime=Clock::now())<timeout_time);
                return m_cb.size();
            }
            else
            {
                m_greatestSignal = singleSemaphoreWait(m_greatestLatch,timeout_time);
                return for_each_popping<false>(constructNonBailing());
            }
        }

        // The default behaviour of the underlying event handler is to wait for all events in its destructor.
        // This will naturally cause you problems if you add functions latched on values you never signal,
        // such as when you change your mind whether to submit. This method is then helpful to avoid a deadlock.
        inline uint32_t abortOldest(const uint64_t upTo)
        {
            return for_each_popping([&](FunctorValuePair& p) -> bool
                {
                    if (p.geSemaValue>upTo)
                        return false;
                    // don't want weird behaviour, so execute everything that would have been executed
                    // if a single `poll()` was called before `abortOldest`
                    if (p.geSemaValue<=m_greatestSignal)
                        p.func();
                    return true;
                }
            );
        }
        inline uint32_t abortLatest(const uint64_t from)
        {
            // We also need to run the functors in the same order they'd be ran with a single `poll()`,
            // so we run all of them from the front, not just from the `from` value.
            for_each_popping(constructNonBailing());
            // now kill the latest stuff
            while (!m_cb.empty() && m_cb.back().geSemaValue>=from)
                m_cb.pop_back();
            return m_cb.size();
        }
        inline void abortAll() {abortOldest(~0ull);}
};

//
template<typename Functor>
class MultiTimelineEventHandlerST final : core::Unmovable, core::Uncopyable
{
    public:
        using TimelineEventHandler = TimelineEventHandlerST<Functor>;
        inline ~MultiTimelineEventHandlerST()
        {
            clear();
        }

        inline const auto& getTimelines() const {return m_timelines;}

        // all the members are counteparts of the single timeline version
        inline uint32_t count() const
        {
            uint32_t sum = 0;
            for (auto p : m_timelines)
                sum += p->count();
            return sum;
        }

        inline void latch(ISemaphore* sema, const uint64_t geValue, Functor&& function)
        {
            auto found = m_timelines.find(sema);
            if (found==m_timelines.end())
            {
                STimeline newTimeline = {
                    .handler = new TimelineEventHandler(core::smart_refctd_ptr<ISemaphore>(sema)),
                    .waitInfoIx = m_scratchWaitInfos.size()
                };
                found = m_timelines.insert(found,std::move(newTimeline));
                m_scratchWaitInfos.emplace_back(sema,0xdeadbeefBADC0FFEull);
            }
            assert(found->handler->getSemaphore()==sema);
            found->handler->latch(sema,geValue,std::move(function));
        }

        inline uint32_t poll()
        {
            uint32_t sum = 0;
            for (auto p : m_timelines)
            {
                bool bailed;
                p->poll(bailed);
                if (bailed)
                    break;
            }
            return sum;
        }

#if 0
        template<class Clock, class Duration=typename Clock::duration>
        inline uint32_t wait(const std::chrono::time_point<Clock,Duration>& timeout_time)
        {
            return 455;
        }
#endif

        inline void abortAll()
        {
            for (auto& p : m_timelines)
                p.handler->abortAll();
            clear();
        }
        inline uint32_t abortOldest(const uint64_t upTo=~0ull)
        {
            uint32_t sum = 0;
            for (auto& p : m_timelines)
                sum += p.handler->abortOldest(upTo);
            return sum;
        }
        inline uint32_t abortLatest(const uint64_t from=0ull)
        {
            uint32_t sum = 0;
            for (auto& p : m_timelines)
                sum += p.handler->abortLatest(from);
            return sum;
        }

    private:
        struct STimeline
        {
            inline auto operator<=>(const STimeline& rhs) const
            {
                return handler->getSemaphore()-rhs.handler->getSemaphore();
            }
            inline auto operator<=>(const ISemaphore* rhs) const
            {
                return handler->getSemaphore()-rhs;
            }

            TimelineEventHandler* handler;
            size_t waitInfoIx;
        };
        // We use a `set<>` instead of `unordered_set<>` because we assume you won't spam semaphores/timelines
        using container_t = core::set<STimeline>;

        template<typename Lambda>
        inline uint32_t for_each_erasing(Lambda&& l)
        {
            uint32_t sum = 0;
            // we don't check erasing when l(*it)==false on purpose, it only happens in poll and the timeline semaphore is likely to get re-added
            for (auto it=m_timelines.begin(); it!=m_timelines.end() && l(*it); )
                it = it->handler->count() ? (it++):eraseTimeline(it);
            return sum;
        }

        inline container_t::iterator eraseTimeline(container_t::iterator timeline)
        {
            // if not the last in scratch
            if (timeline->waitInfoIx<m_scratchWaitInfos.size())
            {
                // swap the mapping with the end scratch element
                const auto& lastScratch = m_scratchWaitInfos.back();
                m_timelines[lastScratch.semaphore].waitInfoIx = timeline->waitInfoIx;
                m_scratchWaitInfos[timeline->waitInfoIx] = lastScratch;
            }
            m_scratchWaitInfos.pop_back();
            delete timeline->handler;
            return m_timelines.erase(timeline);
        }

        inline void clear()
        {
            m_scratchWaitInfos.clear();
            for (auto p : m_timelines)
                delete p.handler;
            m_timelines.clear();
        }

        container_t m_timelines;
        core::vector<ISemaphore::SWaitInfo> m_scratchWaitInfos;
};

}
#endif