#ifndef __NBL_I_CANCELLABLE_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__
#define __NBL_I_CANCELLABLE_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__

#include "nbl/system/IAsyncQueueDispatcher.h"
#include "nbl/system/SReadWriteSpinLock.h"

namespace nbl::system
{


namespace impl
{

class ICancellableAsyncQueueDispatcherBase
{
    public:
        class future_base_t;

        struct request_base_t : impl::IAsyncQueueDispatcherBase::request_base_t
        {
                // do NOT allow canceling of request while they are processed
                bool wait_for_work()
                {
                    uint32_t expected = ES_PENDING;
                    while (!state.compare_exchange_strong(expected,ES_EXECUTING))
                    {
                        // this will allow worker to proceed, but the request predicate will take care of not doing the work
                        if (expected==ES_INITIAL)
                            return false;
                        state.wait(expected);
                        expected = ES_PENDING;
                    }
                    assert(expected==ES_PENDING);
                    return true;
                }
                // to call after request is done being processed
                void notify_ready()
                {
                    const auto prev = state.exchange(ES_READY);
                    assert(prev==ES_INITIAL||prev==ES_EXECUTING);
                    state.notify_one();
                }

            private:
                friend future_base_t;
                friend ICancellableAsyncQueueDispatcherBase;

                //! Atomically cancels this request
                bool set_cancel();
                bool query_cancel() const
                {
                    return state.load()==ES_INITIAL||future==nullptr;
                }

                future_base_t* future = nullptr;

                //! See ICancellableAsyncQueueDispatcher::associate_request_with_future() docs
                void associate_future_object(future_base_t* _future)
                {
                    future = _future;
                }
        };

        class future_base_t
        {
                friend request_base_t;

            protected:
                // this tells us whether and object with a lifetime has been constructed over the memory backing the future
                std::atomic_bool valid_flag = false;
                std::atomic<request_base_t*> request = nullptr;

                // the base class is not directly usable
                future_base_t() = default;
                // future_t is non-copyable and non-movable
                future_base_t(const future_base_t&) = delete;
                // the base class shouldn't be used by itself without casting
                ~future_base_t()
                {
                    // derived should call its own `cancel` in the destructor, I'm just checking its already done
                    assert(!(request.load()||valid_flag.load()));
                }

                bool cancel()
                {
                    request_base_t* req = request.exchange(nullptr);
                    if (req)
                        return req->set_cancel();
                    return false;
                }

            public:

                // Misused these a couple times so i'll better put [[nodiscard]] here 
                [[nodiscard]] bool ready() const 
                { 
                    request_base_t* req = request.load();
                    return !req || req->state.load()==impl::IAsyncQueueDispatcherBase::request_base_t::ES_READY;
                }
                [[nodiscard]] bool valid() const { return valid_flag.load(); }

                void wait()
                {
                    // the request is backed by a circular buffer, so if there was a pointer, it will stay valid (but request might have been overwritten)
                    request_base_t* req = request.load();
                    // all the data is stored inside the future during the request execution, so we dont need access to the request struct after its done executing 
                    // could have used wait_ready() && discard_storate() but its more efficient that way
                    if (req)
                        req->transition(impl::IAsyncQueueDispatcherBase::request_base_t::ES_READY,impl::IAsyncQueueDispatcherBase::request_base_t::ES_INITIAL);
                }
        };

    protected:
        template<class FutureType>
        FutureType* request_get_future_object(request_base_t& req_base)
        {
            return static_cast<FutureType*>(req_base.future);
        }
        void request_associate_future_object(request_base_t& req, future_base_t* future)
        {
            req.associate_future_object(future);
        }
};

}


template <typename CRTP, typename RequestType, uint32_t BufferSize = 256u, typename InternalStateType = void>
class ICancellableAsyncQueueDispatcher : public IAsyncQueueDispatcher<CRTP, RequestType, BufferSize, InternalStateType>, public impl::ICancellableAsyncQueueDispatcherBase
{
        using this_async_queue_t = ICancellableAsyncQueueDispatcher<CRTP, RequestType, BufferSize, InternalStateType>;
        using base_t = IAsyncQueueDispatcher<CRTP, RequestType, BufferSize, InternalStateType>;
        friend base_t;

        template <typename T>
        class future_storage_t
        {
            public:
                alignas(T) uint8_t storage[sizeof(T)];

                T* getStorage() { return reinterpret_cast<T*>(storage); }
        };

    public:
        using request_base_t = impl::ICancellableAsyncQueueDispatcherBase::request_base_t;

        static_assert(std::is_base_of_v<request_base_t, RequestType>, "Request type must derive from request_base_t!");

        template <typename T>
        class future_t : private future_storage_t<T>, public impl::ICancellableAsyncQueueDispatcherBase::future_base_t
        {
                friend this_async_queue_t;

            protected:
                // construct the retval element 
                template <typename... Args>
                void notify(Args&&... args)
                {
                    new (future_storage_t<T>::getStorage()) T(std::forward<Args>(args)...);
                    valid_flag.store(true);
                }

                //! See ICancellableAsyncQueueDispatcher::associate_request_with_future() docs
                void associate_request(RequestType* req)
                {
                    request = req;
                }

            public:
                using value_type = T;

                bool cancel()
                {
                    const bool retval = impl::ICancellableAsyncQueueDispatcherBase::future_base_t::cancel();
                    bool valid = valid_flag.exchange(false);
                    if (valid)
                        future_storage_t<T>::getStorage()->~T();
                    return retval;
                }

                future_t() = default;

                ~future_t()
                {
                    const bool didntUseFuture = cancel();
                    _NBL_DEBUG_BREAK_IF(didntUseFuture);
                }

                T& get()
                {
                    future_base_t::wait();
                    assert(valid_flag);
                    T* ptr = future_storage_t<T>::getStorage();
                    return ptr[0];
                }
        };

        using base_t::base_t;

    protected:
        //! Must be called from within process_request()
        //! User is responsible for providing a value into the associated future object
        template <typename T, typename... Args>
        void notify_future(RequestType& req, Args&&... args)
        {
            request_get_future_object<future_t<T> >(req)->notify(std::forward<Args...>(args)...);
        }

        //! Must be called from within request_impl()
        //! User is responsible for associating future object with a request
        //! Request is automatically cancelled if it is not associated with any future object
        //! More than one request associated with the same future object is undefined behaviour
        template <typename T>
        void associate_request_with_future(RequestType& req, future_t<T>& future)
        {
            assert(!future.valid());
            impl::ICancellableAsyncQueueDispatcherBase::request_associate_future_object(req,&future);
            future.associate_request(&req);
        }
};

// returns false if we haven't cancelled the request in time before it was executed
inline bool impl::ICancellableAsyncQueueDispatcherBase::request_base_t::set_cancel()
{
    // double cancellation
    if (!future)
        return false;
    // wait in case of processing
    uint32_t expected = ES_PENDING;
    while (!state.compare_exchange_strong(expected,ES_INITIAL))
    {
        if (expected==ES_READY)
        {
            transition(ES_READY,ES_INITIAL);
            return true;
        }
        else if (expected==ES_INITIAL) // cancel after await
        {
            return false;
        }
        // was executing, we didnt get here on time
        state.wait(expected);
        expected = ES_PENDING;
    }
    // we've actually cancelled a pending request, and need to cleanup the future
    if (future)
        future->request = nullptr;
    future = nullptr;
    return true;
}

}

#endif
