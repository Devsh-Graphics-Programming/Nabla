#ifndef __NBL_I_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__
#define __NBL_I_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__

#include <atomic>
#include "nbl/core/declarations.h"
#include "nbl/system/IThreadHandler.h"

namespace nbl::system
{

namespace impl
{
    class IAsyncQueueDispatcherBase
    {
    public:
        IAsyncQueueDispatcherBase() = default;
        ~IAsyncQueueDispatcherBase() = default;
        // dont want to play around with relaxed memory ordering yet
        struct request_base_t
        {
                enum E_STATE : uint32_t
                {
                    ES_INITIAL=0,
                    ES_RECORDING=1,
                    ES_PENDING=2,
                    ES_EXECUTING=3,
                    ES_READY=4
                };
                request_base_t() : state(ES_INITIAL)
                {
                    // @devshgraphicsprogramming i assume youre writing in relwithdebuginfo so this doesn't throw an error for you
                    //assert(std::atomic_uint32_t::is_lock_free());
                }
                ~request_base_t() = default;

                // lock when overwriting the request
                void reset()
                {
                    transition(ES_INITIAL,ES_RECORDING);
                }
                // unlock request after we've written everything into it
                void finalize()
                {
                    const auto prev = state.exchange(ES_PENDING);
                    assert(prev==ES_RECORDING);
                    state.notify_one();
                }
                // do NOT allow canceling of request while they are processed
                void wait_for_work()
                {
                    transition(ES_PENDING,ES_EXECUTING);
                }
                // to call after request is done being processed
                void notify_ready()
                {
                    const auto prev = state.exchange(ES_READY);
                    assert(prev==ES_EXECUTING);
                    state.notify_one();
                }
                // to call to await the request to finish processing
                void wait_ready()
                {
                    wait_for(ES_READY);
                }
                // to call after done reading the request and its memory can be recycled
                void discard_storage()
                {
                    const auto prev = state.exchange(ES_INITIAL);
                    assert(prev==ES_READY);
                    state.notify_one();
                }

            protected:
                void transition(const E_STATE from, const E_STATE to)
                {
                    uint32_t expected = from;
                    while (!state.compare_exchange_strong(expected,to))
                    {
                        state.wait(expected);
                        expected = from;
                    }
                    assert(expected==from);
                }
                void wait_for(const E_STATE waitVal)
                {
                    uint32_t current;
                    while ((current=state.load())!=waitVal)
                        state.wait(current);
                }

                std::atomic_uint32_t state;
        };
    };
}

/**
* Provided RequestType shall define 5 methods:
* T reset();
* void finalize(T&&);
* T wait_for_work();
* T wait_for_result();
* T notify_all_ready(T&&);
* TODO: [outdated docs] lock() will be called just before processing the request, and unlock() will be called just after processing the request.
* Those are to enable safe external write access to the request struct for user-defined purposes.
*
* wait_for_result() will wait until the Async queue completes processing the request and notifies us that the request is ready,
* the request will remain locked upon return (so nothing overwrites its address on the circular buffer)
* 
* notify_all_ready() takes an r-value reference to an already locked mutex and notifies any waiters then releases the lock
*/
template <typename CRTP, typename RequestType, uint32_t BufferSize = 256u, typename InternalStateType = void>
class IAsyncQueueDispatcher : public IThreadHandler<CRTP, InternalStateType>, public impl::IAsyncQueueDispatcherBase
{
        static_assert(std::is_base_of_v<impl::IAsyncQueueDispatcherBase::request_base_t, RequestType>, "Request type must derive from request_base_t!");
        static_assert(BufferSize>0u, "BufferSize must not be 0!");
        static_assert(core::isPoT(BufferSize), "BufferSize must be power of two!");

    protected:
        using base_t = IThreadHandler<CRTP, InternalStateType>;
        friend base_t;
    private:
        constexpr static inline uint32_t MaxRequestCount = BufferSize;

        using atomic_counter_t = std::atomic_uint64_t;
        using counter_t = atomic_counter_t::value_type;

        RequestType request_pool[MaxRequestCount];
        atomic_counter_t cb_begin = 0u;
        atomic_counter_t cb_end = 0u;

        static inline counter_t wrapAround(counter_t x)
        {
            constexpr counter_t Mask = static_cast<counter_t>(BufferSize) - static_cast<counter_t>(1);
            return x & Mask;
        }


    public:

        IAsyncQueueDispatcher() = default;
        ~IAsyncQueueDispatcher() = default;

        using mutex_t = typename base_t::mutex_t;
        using lock_t = typename base_t::lock_t;
        using cvar_t = typename base_t::cvar_t;
        using internal_state_t = typename base_t::internal_state_t;

        using request_t = RequestType;

        ///////
        // Required accessible methods of class being CRTP parameter:

        //void init(internal_state_t* state); // required only in case of custom internal state

        //void exit(internal_state_t* state); // optional, no `state` parameter in case of no internal state

        //void request_impl(request_t& req, ...); // `...` are parameteres forwarded from request(), the request's state is locked with a mutex during the call
        //bool process_request_predicate(const request_t& req); // optional, always true if not provided
        //void process_request(request_t& req, internal_state_t& state); // no `state` parameter in case of no internal state
        //void background_work() // optional, does nothing if not provided
        ///////

        using base_t::base_t;

        template <typename... Args>
        request_t& request(Args&&... args)
        {
            auto virtualIx = cb_end++;
            auto safe_begin = virtualIx<MaxRequestCount ? static_cast<counter_t>(0) : (virtualIx-MaxRequestCount+1u);

            for (counter_t old_begin; (old_begin = cb_begin.load()) < safe_begin; )
                cb_begin.wait(old_begin);

            const auto r_id = wrapAround(virtualIx);

            request_t& req = request_pool[r_id];
            req.reset();
            static_cast<CRTP*>(this)->request_impl(req, std::forward<Args>(args)...);
            req.finalize();

            {
                auto global_lk = base_t::createLock();
                // wake up queue thread (needs to happen under a lock to not miss a wakeup)
                base_t::m_cvar.notify_one();
            }
            return req;
        }

    protected:
        bool process_request_predicate(const request_t& req)
        {
            return true;
        }

        void background_work() {}
    private:
        template <typename... Args>
        void work(lock_t& lock, Args&&... optional_internal_state)
        {
            lock.unlock();
            static_assert(sizeof...(optional_internal_state) <= 1u, "How did this happen");

            static_cast<CRTP*>(this)->background_work();

            if (cb_begin!=cb_end) // need this for background work to be done via synthetic wakeups but without new requests being placed (like win32 window manager)
            {
                uint64_t r_id = cb_begin;
                r_id = wrapAround(r_id);

                request_t& req = request_pool[r_id];
                // do NOT allow cancelling or modification of the request while working on it
                req.wait_for_work();
                if (static_cast<CRTP*>(this)->process_request_predicate(req))
                {
                    static_cast<CRTP*>(this)->process_request(req, optional_internal_state...);
                }
                // wake the waiter up
                req.notify_ready();
                cb_begin++;
                #if __cplusplus >= 202002L
                    cb_begin.notify_one();
                #endif
            }
            lock.lock();
        }


        bool wakeupPredicate() const { return (cb_begin != cb_end); }
        bool continuePredicate() const { return (cb_begin != cb_end); }
};

}

#endif
