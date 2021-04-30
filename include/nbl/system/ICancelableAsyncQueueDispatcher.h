#ifndef __NBL_I_CANCELABLE_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__
#define __NBL_I_CANCELABLE_ASYNC_QUEUE_DISPATCHER_H_INCLUDED__

#include "nbl/system/IAsyncQueueDispatcher.h"
#include "nbl/system/SReadWriteSpinLock.h"

namespace nbl {
namespace system
{

struct SCancelableRequestBase : impl::IAsyncQueueDispatcherBase::request_base_t
{
    void set_skip_no_lock()
    {
        skip = true;
    }
    void set_skip()
    {
        write_lock_guard<> lk(rwlock);
        set_skip_no_lock();
    }
    bool query_skip() const
    {
        bool val = false;
        {
            read_lock_guard<> lk(rwlock);
            val = skip;
        }
        return val;
    }

    void lock()
    {
        rwlock.lock_write();
    }
    void unlock()
    {
        rwlock.unlock_write();
    }

    mutable SReadWriteSpinLock rwlock;

private:
    bool skip = false;
};

template <typename CRTP, typename RequestType, uint32_t BufferSize = 256u, typename InternalStateType = void>
class ICancelableAsyncQueueDispatcher : public IAsyncQueueDispatcher<CRTP, RequestType, BufferSize, InternalStateType>
{
    using base_t = IAsyncQueueDispatcher<CRTP, RequestType, BufferSize, InternalStateType>;
    friend base_t;

    using request_base_t = SCancelableRequestBase;

    static_assert(std::is_base_of_v<SCancelableRequestBase, RequestType>, "Request type must derive from request_base_t!");

protected:
    bool process_request_predicate(const typename base_t::request_t& req)
    {
        return !req.query_skip();
    }
};

}}

#endif
