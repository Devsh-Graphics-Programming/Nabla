#ifndef _NBL_VIDEO_I_SEMAPHORE_H_INCLUDED_
#define _NBL_VIDEO_I_SEMAPHORE_H_INCLUDED_


#include "nbl/core/IReferenceCounted.h"

#include "nbl/video/decl/IBackendObject.h"


namespace nbl::video
{

class ISemaphore : public IBackendObject
{
    public:
        virtual uint64_t getCounterValue() const = 0;

        //! Basically the counter can only monotonically increase with time (ergo the "timeline"):
        // 1. `value` must have a value greater than the current value of the semaphore (what you'd get from `getCounterValue()`)
        // 2. `value` must be less than the value of any pending semaphore signal operations (this is actually more complicated)
        // Current pending signal operations can complete in any order, unless there's an execution dependency between them,
        // this will change the current value of the semaphore. Consider a semaphore with current value of 2 and pending signals of 3,4,5;
        // without any execution dependencies, you can only signal a value higher than 2 but less than 3 which is impossible.
        virtual void signal(const uint64_t value) = 0;

        // Vulkan: const VkSemaphore*
        virtual const void* getNativeHandle() const = 0;

        //! Flags for imported/exported allocation
        enum E_EXTERNAL_HANDLE_TYPE : uint32_t
        {
            EHT_NONE = 0x00000000,
            EHT_OPAQUE_FD = 0x00000001,
            EHT_OPAQUE_WIN32 = 0x00000002,
            EHT_OPAQUE_WIN32_KMT = 0x00000004,
            EHT_D3D12_FENCE = 0x00000008,
            EHT_SYNC_FD = 0x00000010,
        };

        //!
        struct SCreationParams
        {
            // A Pre-Destroy-Step is called out just before a `vkDestory` or `glDelete`, this is only useful for "imported" resources
            std::unique_ptr<ICleanup> preDestroyCleanup = nullptr;
            // A Post-Destroy-Step is called in this class' destructor, this is only useful for "imported" resources
            std::unique_ptr<ICleanup> postDestroyCleanup = nullptr;
            // Thus the destructor will skip the call to `vkDestroy` or `glDelete` on the handle, this is only useful for "imported" objects
            bool skipHandleDestroy = false;
            // Handle Type for external resources
            core::bitflag<E_EXTERNAL_HANDLE_TYPE> externalHandleTypes = EHT_NONE;
            //! Imports the given handle  if externalHandle != nullptr && externalMemoryHandleType != EHT_NONE
            //! Creates exportable memory if externalHandle == nullptr && externalMemoryHandleType != EHT_NONE
            void* externalHandle = nullptr;

            uint64_t initialValue = 0;
        };

        auto const& getCreationParams() const
        {
            return m_creationParams;
        }

    protected:
        ISemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& dev, SCreationParams&& params = {})
            : IBackendObject(std::move(dev))
            , m_creationParams(std::move(params))
        {}
        virtual ~ISemaphore() = default;

        SCreationParams m_creationParams;
};

class NBL_API2 TimelineEventHandlerBase : core::Unmovable, core::Uncopyable
{
    public:
        // little utility
        inline ISemaphore* getSemaphore() const {return m_sema.get();}

    protected:
        inline TimelineEventHandlerBase(core::smart_refctd_ptr<ISemaphore>&& sema) : m_sema(std::move(sema)), m_greatestSignal(m_sema->getCounterValue()) {}
        
        template<class Clock, class Duration=typename Clock::duration>
        bool singleSemaphoreWait(const uint64_t value, const std::chrono::time_point<Clock,Duration>& timeout_time)
        {
            const auto current_time = Clock::now();
            if (timeout_time>current_time && notTimedOut(value,std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_time-current_time).count());
                return value; // we return it even on device loss or error, as to not hang up blocks for completion
            return m_sema->getCounterValue();
        }

        bool notTimedOut(const uint64_t value, const uint64_t nanoseconds);

        core::smart_refctd_ptr<ISemaphore> m_sema;
        uint64_t m_greatestSignal;
        uint64_t m_greatestLatch;
};

#if 0
// Could be quite easily made MT and relatively lockless, if only had a good lock-poor circular buffer impl
template<typename Functor>
class TimelineEventHandlerST final : public TimelineEventHandlerBase
{
        constexpr static inline bool ReturnsBool = std::is_same_v<decltype(std::declval<Functor>()()),bool>;
        struct FunctorValuePair
        {
            Functor func;
            uint64_t geSemaValue;
        };
        // could be a circular buffer but whatever
        core::deque<FunctorValuePair> m_cb;
        
        inline uint32_t resetLatch()
        {
            m_greatestLatch = m_cb.empty() ? 0:m_cb.back().geSemaValue;
            return m_cb.size();
        }

    public:
        inline TimelineEventHandlerST(core::smart_refctd_ptr<ISemaphore>&& sema, const uint64_t initialCapacity=4095/sizeof(FunctorValuePair)+1) :
            TimelineEventHandlerBase(std::move(sema)), m_cb(initialCapacity)
        {
            resetLatch();
        }
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

        // Returns number of events still outstanding
        inline uint32_t poll(bool& bailed)
        {
            m_greatestSignal = m_sema->getCounterValue();
            // in a threadsafe scenario, you'd immediately pop everything you can with geSemaValue<=signal
            while (!m_cb.empty() && m_cb.front().geSemaValue<=m_greatestSignal)
            {
                bailed = false;
                if constexpr (ReturnsBool)
                    bailed = m_cb.front().func();
                m_cb.pop_front();
                if (bailed)
                    break;
            }
            return resetLatch();
        }
        inline uint32_t poll()
        {
            bool dummy;
            return poll(dummy);
        }

        template<class Clock, class Duration=typename Clock::duration>
        inline uint32_t wait(const std::chrono::time_point<Clock,Duration>& timeout_time)
        {
            if constexpr (ReturnsBool)
            {
                // Perf-assumption: there are no latched events with wait values less or equal to m_greatestSignal
                // So we have a bunch of events with semaphore values between m_greatestSignal and m_greatestLatch
#if 0
                for (std::chrono::time_point<Clock, Duration> currentClockTime; (currentClockTime = Clock::now()) < timeout_time; )
                while (!m_cb.empty() && m_cb.front().geSemaValue<=m_greatestSignal)
                {
                    const bool bail = m_cb.front().func();
                    m_cb.pop_front();
                    if (bail)
                        return resetLatch();
                }
#endif
            }
            else
            {
                m_greatestSignal = singleSemaphoreWait(m_greatestLatch,timeout_time);
                while (!m_cb.empty() && m_cb.front().geSemaValue<=m_greatestSignal)
                {
                    m_cb.front().func();
                    m_cb.pop_front();
                }
            }
            return resetLatch();
        }

        // The default behaviour of the underlying event handler is to wait for all events in its destructor.
        // This will naturally cause you problems if you add functions latched on values you never signal,
        // such as when you change your mind whether to submit. This method is then helpful to avoid a deadlock.
        inline uint32_t abortOldest(const uint64_t upTo=~0ull)
        {
            m_greatestSignal = m_sema->getCounterValue();
            while (!m_cb.empty() && m_cb.front().geSemaValue<=upTo)
            {
                // don't want non-determinitistic behaviour, so execute everything that would have been executed anyway with a while(pollForReady())
                if (m_cb.front().geSemaValue<= m_greatestSignal)
                    m_cb.front().func();
                m_cb.pop_front();
            }
            return resetLatch();
        }
        inline uint32_t abortLatest(const uint64_t from=0ull)
        {
            m_greatestSignal = m_sema->getCounterValue();
            while (!m_cb.empty() && m_cb.back().geSemaValue>=from)
            {
                // don't want non-determinitistic behaviour, so execute everything that would have been executed anyway with a while(pollForReady())
                if (m_cb.back().geSemaValue<= m_greatestSignal)
                    m_cb.back().func();
                m_cb.pop_back();
            }
            return resetLatch();
        }
};

template<typename Functor>
class MultiTimelineEventHandlerST final
{
    public:
        inline ~MultiTimelineEventHandlerST()
        {
            for (auto p : m_timelines)
                delete p;
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
                found m_timelines.insert(found,new TimelineEventHandlerST(core::smart_refctd_ptr<ISemaphore>(sema)));
            assert((*found)->getSemaphore()==sema);
            found->latch(sema,geValue,std::move(function));
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
        template<class Clock, class Duration=typename Clock::duration>
        inline uint32_t wait(const std::chrono::time_point<Clock, Duration>& timeout_time)
        {
            // want to give each event equal wait time, so interpolate (albeit weirdly)
            return 455;
        }

        inline uint32_t abortOldest(const uint64_t upTo=~0ull)
        {
            uint32_t sum = 0;
            for (auto p : m_timelines)
                sum += p->abortOldest(upTo);
            return sum;
        }
        inline uint32_t abortLatest(const uint64_t from=0ull)
        {
            uint32_t sum = 0;
            for (auto p : m_timelines)
                sum += p->abortLatest(from);
            return sum;
        }

    private:
        struct Compare
        {
            inline bool operator()(const TimelineEventHandlerST* lhs, const TimelineEventHandlerST* rhs) const
            {
                return lhs->getSemaphore()<rhs->getSemaphore();
            }
            inline bool operator()(const TimelineEventHandlerST* lhs, const ISemaphore* rhs) const
            {
                return lhs->getSemaphore()<rhs;
            }
        };
        core::set<TimelineEventHandlerST*,Compare> m_timelines;
};
#endif

}
#endif