#ifndef _NBL_I_ASYNC_QUEUE_DISPATCHER_H_INCLUDED_
#define _NBL_I_ASYNC_QUEUE_DISPATCHER_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/system/IThreadHandler.h"
#include "nbl/system/atomic_state.h"

namespace nbl::system
{

namespace impl
{
class IAsyncQueueDispatcherBase
{
    public:
        struct future_base_t;
        // dont want to play around with relaxed memory ordering yet
        struct request_base_t
        {
            public:
                enum class STATE : uint32_t
                {
                    INITIAL=0,
                    RECORDING=1,
                    PENDING=2,
                    EXECUTING=3,
                    CANCELLED=4
                };

                // in case you need it, which you won't
                inline const auto& getState() const {return state;}

                //! REQUESTING THREAD: lock when overwriting the request's data
                inline void start()
                {
                    state.waitTransition(STATE::RECORDING,STATE::INITIAL);
                    // previous thing cleaned up after itself
                    assert(!future);
                }
                //! REQUESTING THREAD: unlock request after we've written everything into it
                void finalize(future_base_t* fut);
                
                //! WORKER THREAD: returns when work is ready, will deadlock if the state will not eventually transition to pending
                [[nodiscard]] bool wait();
                //! WORKER THREAD: to call after request is done being processed, will deadlock if the request was not executed
                void notify();

                //! ANY THREAD [except worker]: via cancellable_future_t::cancel
                inline void cancel()
                {
                    const auto prev = state.exchangeNotify<false>(STATE::CANCELLED);
                    // If we were in EXECUTING then worker thread is definitely stuck in `base_t::disassociate_request` spinlock
                    assert(prev==STATE::PENDING || prev==STATE::EXECUTING);
                    // sanity check, but its not our job to set it to nullptr
                    assert(future);
                }

            protected:
                // the base class is not directly usable
                inline ~request_base_t()
                {
                    // fully cleaned up
                    assert(!future);
                }

                // ban certain operators
                request_base_t(const request_base_t&) = delete;
                request_base_t(request_base_t&&) = delete;
                request_base_t& operator=(const request_base_t&) = delete;
                request_base_t& operator=(request_base_t&&) = delete;
            
            private:
                future_base_t* future = nullptr;
                atomic_state_t<STATE,STATE::INITIAL> state = {};
        };
        // TODO: attempt to fight the virtualism
        struct future_base_t
        {
            public:
                enum class STATE : uint32_t
                {
                    // after constructor, cancellation, move or destructor
                    INITIAL=0,
                    // after submitting request to work queue
                    ASSOCIATED=1,
                    // while being executed
                    EXECUTING=2,
                    // ready for consumption
                    READY=3,
                    // uncancellable
                    LOCKED=4
                };

                //! REQUESTING THREAD: done as part of filling out the request
                virtual inline void associate_request(request_base_t* req)
                {
                    // sanity check
                    assert(req->getState().query()==request_base_t::STATE::RECORDING);
                    // if not initial state then wait until it gets moved, etc.
                    state.waitTransition(STATE::ASSOCIATED,STATE::INITIAL);
                }
                //! WORKER THREAD: done as part of execution at the very start, after we want to begin work
                [[nodiscard]] virtual inline bool disassociate_request()
                {
                    return state.waitAbortableTransition(STATE::EXECUTING,STATE::ASSOCIATED,STATE::INITIAL);
                }
                //! WORKER THREAD: done as part of execution at the very end, after object is constructed
                inline void notify()
                {
                    state.exchangeNotify<true>(STATE::READY,STATE::EXECUTING);
                }

            protected:
                // the base class is not directly usable
                virtual inline ~future_base_t()
                {
                    // non-cancellable future just need to get to this state, and cancellable will move here
                    state.wait([](const STATE _query)->bool{return _query!=STATE::INITIAL;});
                }
                // future_t is non-copyable and non-movable because request needs a pointer to it
                future_base_t(const future_base_t&) = delete;
                future_base_t(future_base_t&&) = delete;
                future_base_t& operator=(const future_base_t&) = delete;
                future_base_t& operator=(future_base_t&&) = delete;

                // this tells us whether an object with a lifetime has been constructed over the memory backing the future
                // also acts as a lock
                atomic_state_t<STATE,STATE::INITIAL> state= {};
        };

    protected:
        IAsyncQueueDispatcherBase() = default;
        ~IAsyncQueueDispatcherBase() = default;

    public:
        template<typename T>
        class future_t : private core::StorageTrivializer<T>, protected future_base_t
        {
                using storage_t = core::StorageTrivializer<T>;
                inline void discard_common()
                {
                    storage_t::destruct();
                    state.exchangeNotify<true>(STATE::INITIAL,STATE::LOCKED);
                }

            public:
                inline future_t() = default;
                inline ~future_t()
                {
                    discard();
                }

                //!
                inline bool ready() const
                {
                    switch (state.query())
                    {
                        case STATE::LOCKED:
                            [[fallthrough]];
                        case STATE::READY:
                            return true;
                        default:
                            break;
                    }
                    return false;
                }
                
                //! Returns after waiting till `ready()` would be true or after 
                inline bool wait() const
                {
                    bool retval = false;
                    state.wait([&retval](const STATE _query)->bool{
                        switch (_query)
                        {
                        case STATE::INITIAL:
                            return false;
                            break;
                        case STATE::READY:
                            [[fallthrough]];
                        case STATE::LOCKED:
                            retval = true;
                            return false;
                            break;
                        default:
                            break;
                        }
                        return true;
                    });
                    return retval;
                }

                //! NOTE: Deliberately named `...acquire` instead of `...lock` to make them incompatible with `unique_lock`
                // and other RAII locks as the blocking aquire can fail and that needs to be handled.

                //! ANY THREAD [except WORKER]: If we're READY transition to LOCKED
                [[nodiscard]] inline T* try_acquire()
                {
                    auto expected = STATE::READY;
                    if (state.tryTransition(STATE::LOCKED,expected))
                        return storage_t::getStorage();
                    return nullptr;
                }
                //! ANY THREAD [except WORKER]: Wait till we're either in READY and move us to LOCKED or bail on INITIAL
                // this accounts for being cancelled or consumed while waiting
                [[nodiscard]] inline T* acquire()
                {
                    if (state.waitAbortableTransition(STATE::LOCKED,STATE::READY,STATE::INITIAL))
                        return storage_t::getStorage();
                    return nullptr;
                }
                //! ANY THREAD [except WORKER]: Release an acquired lock
                inline void release()
                {
                    state.exchangeNotify<true>(STATE::READY,STATE::LOCKED);
                }

                //! NOTE: You're in charge of ensuring future doesn't transition back to INITIAL (e.g. lock or use sanely!)
                inline const T* get() const
                {
                    if (ready())
                        return storage_t::getStorage();
                    return nullptr;
                }
                inline T* get()
                {
                    if (future_base_t::state.query() != future_base_t::STATE::LOCKED)
                        return nullptr;
                    return storage_t::getStorage();
                }

                //! Can only be called once! If returns false means has been cancelled and nothing happened
                inline bool move_into(T& dst)
                {
                    T* pSrc = acquire();
                    if (!pSrc)
                        return false;
                    dst = std::move(*pSrc);
                    discard_common();
                    return true;
                }

                //!
                virtual inline void discard()
                {
                    if (acquire())
                        discard_common();
                }

            protected:
                // construct the retval element 
                template <typename... Args>
                inline void notify(Args&&... args)
                {
                    storage_t::construct(std::forward<Args>(args)...);
                    future_base_t::notify();
                }
        };
        template<typename T>
        struct cancellable_future_t final : public future_t<T>
        {
                using base_t = future_t<T>;
                std::atomic<request_base_t*> request = nullptr;

            public:
                //! ANY THREAD [except WORKER]: Cancel pending request if we can, returns whether we actually managed to cancel
                inline bool cancel()
                {
                    auto expected = base_t::STATE::ASSOCIATED;
                    if (state.tryTransition(STATE::EXECUTING,expected))
                    {
                        // Since we're here we've managed to move from ASSOCIATED to fake "EXECUTING" this means that the Request is either:
                        // 1. RECORDING but after returning from `base_t::associate_request`
                        while (!request.load()) {}
                        // 2. PENDING
                        // 3. EXECUTING but before returning from `base_t::disassociate_request` cause there's a spinlock there
                        
                        request.exchange(nullptr)->cancel();

                        // after doing everything, we can mark ourselves as cleaned up
                        state.exchangeNotify<false>(STATE::INITIAL,STATE::EXECUTING);
                        return true;
                    }
                    // we're here because either:
                    // - there was no work submitted
                    // - someone else cancelled
                    // - request is currently executing
                    // - request is ready
                    // - storage is locked/acquired
                    // sanity check (there's a tiny gap between transitioning to EXECUTING and disassociating request)
                    assert(expected==STATE::EXECUTING || request==nullptr);
                    return false;
                }

                inline void discard() override final
                {
                    // try to cancel
                    cancel();
                    // sanity check
                    assert(request==nullptr);
                    // proceed with the usual
                    base_t::discard();
                }

            private:
                inline void associate_request(request_base_t* req) override
                {
                    base_t::associate_request(req);
                    request_base_t* prev = request.exchange(req);
                    // sanity check
                    assert(prev==nullptr);
                }
                inline bool disassociate_request() override
                {
                    if (base_t::disassociate_request())
                    {
                        // only assign if we didn't get cancelled mid-way, otherwise will mess up `associate_request` sanity checks
                        request_base_t* prev = request.exchange(nullptr);
                        assert(prev && prev->getState().query()==request_base_t::STATE::EXECUTING);
                        return true;
                    }
                    return false;
                }
        };
};

inline void IAsyncQueueDispatcherBase::request_base_t::finalize(future_base_t* fut)
{
    future = fut;
    future->associate_request(this);
    state.exchangeNotify<false>(STATE::PENDING,STATE::RECORDING);
}

inline bool IAsyncQueueDispatcherBase::request_base_t::wait()
{
    if (state.waitAbortableTransition(STATE::EXECUTING,STATE::PENDING,STATE::CANCELLED) && future->disassociate_request())
        return true;
    //assert(future->cancellable);
    future = nullptr;
    state.exchangeNotify<false>(STATE::INITIAL,STATE::CANCELLED);
    return false;
}
inline void IAsyncQueueDispatcherBase::request_base_t::notify()
{
    future->notify();
    // cleanup
    future = nullptr;
    // allow to be recycled
    state.exchangeNotify<false>(STATE::INITIAL,STATE::EXECUTING);
}

}

/**
* Required accessible methods of class being CRTP parameter:
* 
* void init(internal_state_t* state); // required only in case of custom internal state
*
* void exit(internal_state_t* state); // optional, no `state` parameter in case of no internal state
*
* void request_impl(request_t& req, ...); // `...` are parameteres forwarded from request(), the request's state is locked with a mutex during the call
* void process_request(request_t& req, internal_state_t& state); // no `state` parameter in case of no internal state
* void background_work() // optional, does nothing if not provided
* 
* 
* Provided RequestType shall define 5 methods:
* void start();
* void finalize();
* bool wait();
* void notify();
* TODO: [outdated docs] lock() will be called just before processing the request, and unlock() will be called just after processing the request.
* Those are to enable safe external write access to the request struct for user-defined purposes.
*
* wait_for_result() will wait until the Async queue completes processing the request and notifies us that the request is ready,
* the request will remain locked upon return (so nothing overwrites its address on the circular buffer)
* 
* notify_all_ready() takes an r-value reference to an already locked mutex and notifies any waiters then releases the lock
*/
template <typename CRTP, typename RequestType, uint32_t BufferSize = 256u, typename InternalStateType = void>
class IAsyncQueueDispatcher : public IThreadHandler<CRTP,InternalStateType>, protected impl::IAsyncQueueDispatcherBase
{
        static_assert(std::is_base_of_v<impl::IAsyncQueueDispatcherBase::request_base_t,RequestType>, "Request type must derive from request_base_t!");
        static_assert(BufferSize>0u, "BufferSize must not be 0!");
        static_assert(core::isPoT(BufferSize), "BufferSize must be power of two!");

    protected:
        using base_t = IThreadHandler<CRTP,InternalStateType>;
        friend base_t; // TODO: remove, some functions should just be protected

    private:
        constexpr static inline uint32_t MaxRequestCount = BufferSize;

        // maybe one day we'll abstract this into a lockless queue
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
        inline IAsyncQueueDispatcher() {}
        inline IAsyncQueueDispatcher(base_t::start_on_construction_t) : base_t(base_t::start_on_construction_t) {}

        using mutex_t = typename base_t::mutex_t;
        using lock_t = typename base_t::lock_t;
        using cvar_t = typename base_t::cvar_t;
        using internal_state_t = typename base_t::internal_state_t;

        using request_t = RequestType;

        // Returns a reference to a request's storage in the circular buffer after processing the moved arguments
        // YOU MUST CONSUME THE REQUEST by calling `discard_storage()` on it EXACTLY ONCE!
        // YOU MUST CALL IT EVEN IF THERE'S NO DATA YOU WISH TO GET BACK FROM IT!
        // (if you don't the queue will deadlock because of an unresolved overflow)
        template <typename... Args>
        request_t& request(Args&&... args)
        {
            auto virtualIx = cb_end++;
            auto safe_begin = virtualIx<MaxRequestCount ? static_cast<counter_t>(0) : (virtualIx-MaxRequestCount+1u);

            for (counter_t old_begin; (old_begin = cb_begin.load()) < safe_begin; )
                cb_begin.wait(old_begin);

            const auto r_id = wrapAround(virtualIx);

            request_t& req = request_pool[r_id];
            req.start();
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
        inline ~IAsyncQueueDispatcher() {}
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
                if (req.wait_for_work())
                {
                    // if the request supports cancelling and got cancelled, the wait_for_work function may return false
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

        inline bool wakeupPredicate() const { return (cb_begin != cb_end); }
        inline bool continuePredicate() const { return (cb_begin != cb_end); }
};

}

#endif
