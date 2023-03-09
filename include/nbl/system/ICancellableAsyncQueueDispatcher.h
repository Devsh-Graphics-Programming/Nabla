#ifndef _NBL_I_CANCELLABLE_ASYNC_QUEUE_DISPATCHER_H_INCLUDED_
#define _NBL_I_CANCELLABLE_ASYNC_QUEUE_DISPATCHER_H_INCLUDED_

#include "nbl/system/IAsyncQueueDispatcher.h"
#include "nbl/system/SReadWriteSpinLock.h"

namespace nbl::system
{


template <typename CRTP, typename RequestType, uint32_t BufferSize = 256u, typename InternalStateType = void>
class ICancellableAsyncQueueDispatcher : public IAsyncQueueDispatcher<CRTP, RequestType, BufferSize, InternalStateType>, public impl::ICancellableAsyncQueueDispatcherBase
{
        using this_async_queue_t = ICancellableAsyncQueueDispatcher<CRTP, RequestType, BufferSize, InternalStateType>;
        using base_t = IAsyncQueueDispatcher<CRTP, RequestType, BufferSize, InternalStateType>;
        friend base_t;

    public:
        using request_base_t = impl::ICancellableAsyncQueueDispatcherBase::request_base_t;

        static_assert(std::is_base_of_v<request_base_t, RequestType>, "Request type must derive from request_base_t!");

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

}

#endif
