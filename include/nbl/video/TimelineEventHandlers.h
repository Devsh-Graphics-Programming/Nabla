#ifndef _NBL_VIDEO_TIMELINE_EVENT_HANDLERS_H_INCLUDED_
#define _NBL_VIDEO_TIMELINE_EVENT_HANDLERS_H_INCLUDED_


#include "nbl/video/ILogicalDevice.h"

#include <chrono>


namespace nbl::video
{
class TimelineEventHandlerBase : core::Unmovable, core::Uncopyable
{
    public:
        struct PollResult
        {
            uint32_t eventsLeft = ~0u;
            bool bailed = false;
        };

        // little utility
        inline const ISemaphore* getSemaphore() const { return m_sema.get(); }
        
        // todo: rename to default_wait_point ?
        template<class Clock=std::chrono::steady_clock>
        static inline Clock::time_point default_wait()
        {
            return Clock::now()+std::chrono::microseconds(50);
        }

    protected:
        TimelineEventHandlerBase(core::smart_refctd_ptr<const ISemaphore>&& sema) : m_sema(std::move(sema)) {}

        core::smart_refctd_ptr<const ISemaphore> m_sema;
};

// Could be made MT and relatively lockless, if only had a good lock-few circular buffer impl
// Not sure its worth the effort as anything using this will probably need to be lockful to be MT
template<typename Functor>
class TimelineEventHandlerST final : public TimelineEventHandlerBase
{
    public:
        // Theoretically could make a factory function cause passing a null semaphore is invalid, but counting on users to be relatively intelligent.
        inline TimelineEventHandlerST(core::smart_refctd_ptr<const ISemaphore>&& sema, const uint64_t initialCapacity=4095/sizeof(FunctorValuePair)+1) :
            TimelineEventHandlerBase(std::move(sema)), m_greatestLatch(0), m_greatestSignal(m_sema->getCounterValue()) {}
        // If you don't want to deadlock here, look into the `abort*` family of methods
        ~TimelineEventHandlerST()
        {
            while (wait(std::chrono::steady_clock::now()+std::chrono::seconds(5))) {}
        }

        inline uint32_t count() const {return m_cb.size();}

        // You can latch arbitrary functors upon the semaphore reaching some value
        inline void latch(const uint64_t geSemaValue, Functor&& function)
        {
            //const auto oldValue = core::atomic_fetch_max(&m_greatestLatch,geSemaValue);
            assert(geSemaValue>=m_greatestLatch); // you cannot latch out of order
            m_greatestLatch = geSemaValue;
            m_cb.emplace_back(std::move(function),geSemaValue);
        }

        //
        template<typename... Args>
        inline PollResult poll(Args&&... args)
        {
            return poll_impl<true>(std::forward<Args>(args)...);
        }

        template<class Clock, class Duration=typename Clock::duration, typename... Args>
        inline uint32_t wait(const std::chrono::time_point<Clock,Duration>& timeout_time, Args&&... args)
        {
            if (m_cb.empty())
                return 0;

            auto singleSemaphoreWait = [&](const uint64_t waitVal, const std::chrono::time_point<Clock,Duration>& waitPoint) -> void
            {
                // remeber that latch can move back, not signal though
                if (waitVal<=m_greatestSignal)
                    return;

                const auto current_time = Clock::now();
                if (waitPoint>current_time)
                {
                    auto device = const_cast<ILogicalDevice*>(m_sema->getOriginDevice());
                    const auto nanosecondsLeft = std::chrono::duration_cast<std::chrono::nanoseconds>(waitPoint-current_time).count();
                    const ISemaphore::SWaitInfo info = {.semaphore=m_sema.get(),.value = waitVal};
                    if (device->waitForSemaphores({&info,1},true,nanosecondsLeft)==ISemaphore::WAIT_RESULT::SUCCESS)
                    {
                        m_greatestSignal = waitVal;
                        return;
                    }
                }
                m_greatestSignal = m_sema->getCounterValue();
            };
            
            constexpr bool ReturnsBool = std::is_same_v<decltype(std::declval<Functor>()(std::forward<Args>(args)...)),bool>;
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
                    singleSemaphoreWait(m_cb.front().geSemaValue,singleWaitTimePt);

                    bool bailed = false;
                    for_each_popping<false>(constructBailing(bailed,std::forward<Args>(args)...));
                    if (bailed)
                        break;
                } while ((currentTime=Clock::now())<timeout_time);
                return m_cb.size();
            }
            else
            {
                singleSemaphoreWait(m_greatestLatch,timeout_time);
                return for_each_popping<false>(constructNonBailing(std::forward<Args>(args)...));
            }
        }

        // The default behaviour of the underlying event handler is to wait for all events in its destructor.
        // This will naturally cause you problems if you add functions latched on values you never signal,
        // such as when you change your mind whether to submit. This method is then helpful to avoid a deadlock.
        template<typename... Args>
        inline uint32_t abortOldest(const uint64_t upTo, Args&&... args)
        {
            return for_each_popping([&](FunctorValuePair& p) -> bool
                {
                    if (p.geSemaValue>upTo)
                        return false;
                    // don't want weird behaviour, so execute everything that would have been executed
                    // if a single `poll()` was called before `abortOldest`
                    if (p.geSemaValue<=m_greatestSignal)
                        p.func(std::forward<Args>(args)...);
                    return true;
                }
            );
        }
        template<typename... Args>
        inline uint32_t abortLatest(const uint64_t from, Args&&... args)
        {
            // We also need to run the functors in the same order they'd be ran with a single `poll()`,
            // so we run all of them from the front, not just from the `from` value.
            for_each_popping(constructNonBailing(std::forward<Args>(args)...));
            // now kill the latest stuff
            while (!m_cb.empty() && m_cb.back().geSemaValue>=from)
                m_cb.pop_back();
            return m_cb.size();
        }
        template<typename... Args>
        inline void abortAll(Args&&... args) {abortOldest(~0ull,std::forward<Args>(args)...);}

    private:
        // To get access to almost everything
        template<typename Functor_, bool RefcountTheDevice> friend class MultiTimelineEventHandlerST;

        struct FunctorValuePair
        {
            Functor func;
            uint64_t geSemaValue;
        };
        // could be a circular buffer but whatever for now
        core::deque<FunctorValuePair> m_cb;
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

        template<typename... Args>
        inline auto constructNonBailing(Args&&... args)
        {
            return [&](FunctorValuePair& p) -> bool
            {
                if (p.geSemaValue>m_greatestSignal)
                    return false;
                p.func(std::forward<Args>(args)...);
                return true;
            };
        }
        template<typename... Args>
        inline auto constructBailing(bool& bailed, Args&&... args)
        {
            return [&](FunctorValuePair& p) -> bool
            {
                if (bailed || p.geSemaValue>m_greatestSignal)
                    return false;
                const bool bailedBefore = bailed;
                bailed = p.func(std::forward<Args>(args)...);
                return !bailedBefore;
            };
        }

        template<bool QueryCounter, typename... Args>
        inline PollResult poll_impl(Args&&... args)
        {
            PollResult retval = {};
            constexpr bool ReturnsBool = std::is_same_v<decltype(std::declval<Functor>()(std::forward<Args>(args)...)),bool>;
            if constexpr (ReturnsBool)
                retval.eventsLeft = for_each_popping<QueryCounter>(constructBailing(retval.bailed, std::forward<Args>(args)...));
            else
                retval.eventsLeft = for_each_popping<QueryCounter>(constructNonBailing(std::forward<Args>(args)...));
            return retval;
        }
};

// `RefcountTheDevice` should be false for any "internal user" of the Handler inside the Logical Device, such as the IQueue to avoid circular refs
/*
template<bool RefcountTheDevice>
class MultiTimelineEventHandlerBase : core::Unmovable, core::Uncopyable
{
    public:
        using device_ptr_t = std::conditional_t<RefcountTheDevice,core::smart_refctd_ptr<ILogicalDevice>,ILogicalDevice*>;
        inline MultiTimelineEventHandlerBase(device_ptr_t&& device) : m_device(std::move(device)) {}

        inline ILogicalDevice* getLogicalDevice() const {return m_device.get();}

    protected:
        template<typename Functor>
        static inline auto getGreatestSignal(const TimelineEventHandlerST<Functor>* handler) {return handler->m_greatestSignal;}
        template<typename Functor>
        static inline auto getEmpty(const TimelineEventHandlerST<Functor>* handler) {return handler->m_cb.empty();}

        device_ptr_t m_device;
};
*/
template<typename Functor, bool RefcountTheDevice=true>
class MultiTimelineEventHandlerST final : core::Unmovable, core::Uncopyable
{
    public:
        using TimelineEventHandler = TimelineEventHandlerST<Functor>;
        using device_ptr_t = std::conditional_t<RefcountTheDevice,core::smart_refctd_ptr<ILogicalDevice>,ILogicalDevice*>;

        inline MultiTimelineEventHandlerST(ILogicalDevice* device) : m_device(device) {}
        MultiTimelineEventHandlerST(const MultiTimelineEventHandlerST&) = delete;
        inline ~MultiTimelineEventHandlerST()
        {
            clear();
        }

        //
        MultiTimelineEventHandlerST& operator=(const MultiTimelineEventHandlerST&) = delete;
        
        //
        inline ILogicalDevice* getLogicalDevice() const
        {
            if constexpr (RefcountTheDevice)
                return m_device.get();
            else
                return m_device;
        }

        //
        inline const auto& getTimelines() const {return m_timelines;}

        // all the members are counteparts of the single timeline version
        inline uint32_t count() const
        {
            uint32_t sum = 0;
            for (auto p : m_timelines)
                sum += p.handler->count();
            return sum;
        }

        inline bool latch(const ISemaphore::SWaitInfo& futureWait, Functor&& function)
        {
            auto found = m_timelines.find(futureWait.semaphore);
            if (found==m_timelines.end())
            {
                if (futureWait.semaphore->getOriginDevice()!=getLogicalDevice())
                    return false;
                STimeline newTimeline = {
                    .handler = new TimelineEventHandler(core::smart_refctd_ptr<const ISemaphore>(futureWait.semaphore)),
                    .waitInfoIx = m_scratchWaitInfos.size()
                };
                found = m_timelines.insert(found,std::move(newTimeline));
                m_scratchWaitInfos.emplace_back(futureWait.semaphore,0xdeadbeefBADC0FFEull);
            }
            assert(found->handler->getSemaphore()==futureWait.semaphore);
            found->handler->latch(futureWait.value,std::move(function));
            return true;
        }

        template<typename... Args>
        inline typename TimelineEventHandler::PollResult poll(Args&&... args)
        {            
            typename TimelineEventHandler::PollResult retval = {0,false};
            for (auto it=m_timelines.begin(); it!=m_timelines.end(); )
            {
                if (!retval.bailed)
                {
                    const auto local = it->handler->poll(std::forward<Args>(args)...);
                    retval.eventsLeft += local.eventsLeft;
                    retval.bailed = local.bailed;
                }
                if (it->handler->count())
                    it++;
                else
                    it = eraseTimeline(it);
            }
            return retval;
        }

        template<class Clock, class Duration=typename Clock::duration, typename... Args>
        inline uint32_t wait(const std::chrono::time_point<Clock,Duration>& timeout_time, Args&&... args)
        {
            auto nanosecondsLeft = [](const std::chrono::time_point<Clock,Duration>& waitPoint)->uint64_t
            {
                const auto current_time = Clock::now();
                if (current_time>=waitPoint)
                    return 0;
                return std::chrono::duration_cast<std::chrono::nanoseconds>(waitPoint-current_time).count();
            };

            constexpr bool ReturnsBool = std::is_same_v<decltype(std::declval<Functor>()(std::forward<Args>(args)...)),bool>;
            constexpr bool WaitAll = !ReturnsBool;

            uint32_t sum = 0;
            do
            {
                auto uniqueValueEstimate = 1;
                // `waitsToPerform` isn't very conservative, it doesn't mean there are no latched events
                // instead it means that  there is no point waiting with the device on the semaphore
                // because the value we're about to wait for was already attained.
                bool waitsToPerform = false;
                // first gather all the wait values if there's time to even perform a wait
                if (nanosecondsLeft(timeout_time))
                for (auto it=m_timelines.begin(); it!=m_timelines.end(); )
                {
                    // will return 0 for an empty event list
                    const auto waitVal = it->getWaitValue(WaitAll);
                    if (waitVal)
                    {
                        // need to fill all waits anyway even if its redudant
                        m_scratchWaitInfos[it->waitInfoIx].value = waitVal;
                        // remeber that latch can move back, not the signal though
                        if (waitVal>it->handler->m_greatestSignal)
                        {
                            uniqueValueEstimate = core::max(core::min(it->handler->m_cb.size(),it->handler->m_greatestSignal-it->handler->m_greatestLatch),uniqueValueEstimate);
                            waitsToPerform = true;
                        }
                        it++;
                    }
                    else
                        it = eraseTimeline(it);
                }

                bool allReady = false;
                if (waitsToPerform)
                {
                    const std::chrono::time_point<Clock> singleWaitTimePt((Clock::now().time_since_epoch()*(uniqueValueEstimate-1u)+timeout_time.time_since_epoch())/uniqueValueEstimate);
                    if (const auto nano = nanosecondsLeft(WaitAll ? timeout_time:singleWaitTimePt))
                    if (m_device->waitForSemaphores(m_scratchWaitInfos,WaitAll,nano)==ISemaphore::WAIT_RESULT::SUCCESS)
                        allReady = WaitAll || m_scratchWaitInfos.size()==1;
                }
 
                sum = 0;
                bool bailed = false;
                for (auto it=m_timelines.begin(); it!=m_timelines.end(); )
                {
                    auto* handler = it->handler;
                    // only if we waited for all semaphores, we can just set their greatest signal value to the value we awaited
                    handler->m_greatestSignal = allReady ? it->getWaitValue(WaitAll):handler->getSemaphore()->getCounterValue();
                    if (bailed)
                        sum += handler->count();
                    else
                    {
                        const auto local = handler->poll_impl<false>(std::forward<Args>(args)...);
                        bailed = local.bailed;
                        // if don't have any events left, remove the timeline
                        if (local.eventsLeft)
                        {
                            sum += local.eventsLeft;
                            it++;
                        }
                        // but there's a fast path at the end
                        else if (ReturnsBool || !allReady)
                            it = eraseTimeline(it);
                        else
                            it++;
                    }
                }
                // ultra fast path for non-bailing code when everything was covered by a single wait
                if (WaitAll && allReady)
                    clear();
            } while (sum && Clock::now()<timeout_time);
            return sum;
        }

        inline void abortAll()
        {
            for (auto& p : m_timelines)
                p.handler->abortAll();
            clear();
        }

    private:
        struct STimeline
        {
            inline uint64_t getWaitValue(const bool waitAll) const
            {
                if (handler->m_cb.empty())
                    return 0ull;
                // following same assumptions as the single-timeline case
                if (waitAll)
                    return handler->m_greatestLatch;
                else
                    return handler->m_cb.front().geSemaValue;
            }

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
        // also we need to be able to continue iteration after an erasure of a single element
        using container_t = core::set<STimeline,std::less<void>/*quirk of STL*/>;

        inline container_t::iterator eraseTimeline(typename container_t::iterator timeline)
        {
            // if not the last in scratch
            if (timeline->waitInfoIx+1<m_scratchWaitInfos.size())
            {
                // swap the mapping with the end scratch element
                const auto& lastScratch = m_scratchWaitInfos.back();
                typename container_t::iterator found = m_timelines.find(lastScratch.semaphore);
//                found->waitInfoIx = timeline->waitInfoIx;
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
        device_ptr_t m_device;
};

}
#endif