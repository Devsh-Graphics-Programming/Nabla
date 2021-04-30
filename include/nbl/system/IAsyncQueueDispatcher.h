#ifndef __NBL_I_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__
#define __NBL_I_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__

#include "nbl/system/IThreadHandler.h"
#include "nbl/core/Types.h"
#include "nbl/core/math/intutil.h"

namespace nbl {
namespace system
{

namespace impl
{
    class IAsyncQueueDispatcherBase
    {
    public:
        struct request_base_t
        {
            // wait on this for result to be ready
            std::condition_variable cvar;
            bool ready = false;

            void lock() {}
            void unlock() {}
        };
    };
}

/**
* Provided RequestType may optionally define 2 member functions:
* void lock();
* void unlock();
* lock() will be called just before processing the request, and unlock() will be called just after processing the request.
* Those are to enable safe external write access to the request struct for user-defined purposes.
*/
template <typename CRTP, typename RequestType, uint32_t BufferSize = 256u, typename InternalStateType = void>
class IAsyncQueueDispatcher : public IThreadHandler<CRTP, InternalStateType>, public impl::IAsyncQueueDispatcherBase
{
    static_assert(std::is_base_of_v<impl::IAsyncQueueDispatcherBase::request_base_t, RequestType>, "Request type must derive from request_base_t!");
    static_assert(BufferSize>0u, "BufferSize must not be 0!");

    using base_t = IThreadHandler<CRTP, InternalStateType>;
    friend base_t;

    constexpr static inline uint32_t MaxRequestCount = BufferSize;

    RequestType request_pool[MaxRequestCount];
    uint32_t cb_begin = 0u;
    uint32_t cb_end = 0u;

    static inline uint32_t incAndWrapAround(uint32_t x)
    {
        if constexpr (core::isPoT(BufferSize))
            return (x + 1u) & (BufferSize - 1u);
        else
            return (x + 1u) % BufferSize;
    }

public:
    using mutex_t = typename base_t::mutex_t;
    using lock_t = typename base_t::lock_t;
    using cvar_t = typename base_t::cvar_t;
    using internal_state_t = typename base_t::internal_state_t;

    using request_t = RequestType;

    ///////
    // Required accessible methods of class being CRTP parameter:

    //void init(internal_state_t* state); // required only in case of custom internal state

    //void exit(internal_state_t* state); // optional, no `state` parameter in case of no internal state

    //void request_impl(request_t& req, ...); // `...` are parameteres forwarded from request()
    //bool process_request_predicate(const request_t& req); // optional, always true if not provided
    //void process_request(request_t& req, internal_state_t& state); // no `state` parameter in case of no internal state
    ///////

    template <typename... Args>
    request_t& request(Args&&... args)
    {
        //auto lk = createLock();
        //raii_dispatch_handler_t raii_handler(std::move(lk), m_cvar);
        auto raii_handler = base_t::createRAIIDispatchHandler();

        const uint32_t r_id = cb_end;
        cb_end = incAndWrapAround(cb_end);

        request_t& req = request_pool[r_id];
        req.ready = false;
        static_cast<CRTP*>(this)->request_impl(req, std::forward<Args>(args)...);

        return req;
    }

    void waitForRequestCompletion(request_t& req)
    {
        auto lk = base_t::createLock();
        req.cvar.wait(lk, [&req]() -> bool { return req.ready; });

        assert(req.ready);
    }

protected:
    bool process_request_predicate(const request_t& req)
    {
        return true;
    }

private:
    template <typename... Args>
    void work(lock_t& lock, Args&&... optional_internal_state)
    {
        static_assert(sizeof...(optional_internal_state) <= 1u, "How did this happen");

        request_t& req = request_pool[cb_begin];
        cb_begin = incAndWrapAround(cb_begin);

        // unlock global lock when request is being processed
        lock.unlock();

        req.lock();
        if (static_cast<CRTP*>(this)->process_request_predicate(req))
        {
            static_cast<CRTP*>(this)->process_request(req, optional_internal_state...);
        }
        req.unlock();

        // lock before condition change
        // https://stackoverflow.com/questions/4544234/calling-pthread-cond-signal-without-locking-mutex
        lock.lock();
        req.ready = true;
        req.cvar.notify_all(); //notify_one() would do as well, but lets call notify_all() in case something went horribly wrong (theoretically not possible) and multiple threads are waiting for single request
    }

    bool wakeupPredicate() const { return (cb_begin != cb_end); }
    bool continuePredicate() const { return (cb_begin != cb_end); }
};

}
}

#endif
