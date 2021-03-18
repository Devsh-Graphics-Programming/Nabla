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
        };
    };
}

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
    //void process_request(request_t& req, internal_state_t& state); // no `state` parameter in case of no internal state
    ///////

    template <typename... Args>
    request_t& request(Args&&... args)
    {
        //auto lk = createLock();
        //raii_dispatch_handler_t raii_handler(std::move(lk), m_cvar);
        auto raii_handler = createRAIIDispatchHandler();

        const uint32_t r_id = cb_end;
        cb_end = incAndWrapAround(cb_end);

        request_t& req = request_pool[r_id];
        req.ready = false;
        static_cast<CRTP*>(this)->request_impl(req, std::forward<Args>(args)...);

        return req;
    }

    void waitForRequestCompletion(request_t& req)
    {
        auto lk = createLock();
        req.cvar.wait(lk, [&req]() -> bool { return req.ready; });

        assert(req.ready);
    }

private:
    template <typename... Args>
    void work(lock_t& lock, Args&&... optional_internal_state)
    {
        static_assert(sizeof...(optional_internal_state) <= 1u, "How did this happen");

        request_t& req = request_pool[cb_begin];
        cb_begin = incAndWrapAround(cb_begin);

        static_cast<CRTP*>(this)->process_request(req, optional_internal_state...);

        req.ready = true;
        // moving unlock before the switch (but after cb_begin increment) probably wouldnt hurt
        lock.unlock(); // unlock so that notified thread wont immidiately block again
        req.cvar.notify_all(); //notify_one() would do as well, but lets call notify_all() in case something went horribly wrong (theoretically not possible) and multiple threads are waiting for single request
        lock.lock(); // reacquire (must be locked at the exit of this function -- see system::IThreadHandler docs)
    }

    bool wakeupPredicate() const { return (cb_begin != cb_end); }
    bool continuePredicate() const { return (cb_begin != cb_end); }
};

}
}

#endif
