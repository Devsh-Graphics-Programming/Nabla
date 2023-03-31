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
    protected:
        struct future_base_t;
        // dont want to play around with relaxed memory ordering yet
        struct request_base_t // TODO: protect to anyone but inheritor
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
                inline const atomic_state_t<STATE,STATE::INITIAL>& getState() const {return state;}

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
                [[nodiscard]] future_base_t* wait();
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
                inline request_base_t() = default;
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

            protected:
                friend struct request_base_t;
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

                // the base class is not directly usable
                inline future_base_t() = default;
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
                atomic_state_t<STATE,STATE::INITIAL> state = {};
        };

        // not meant for direct usage
        IAsyncQueueDispatcherBase() = default;
        ~IAsyncQueueDispatcherBase() = default;

    public:
        template<typename T>
        class future_t : private core::StorageTrivializer<T>, public future_base_t
        {
                using storage_t = core::StorageTrivializer<T>;
                
                friend class storage_lock_t;

            public:
                inline future_t() : future_base_t() {}
                virtual inline ~future_t()
                {
                    if (auto lock=acquire())
                        lock.discard();
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

                //! NOTE: You're in charge of ensuring future doesn't transition back to INITIAL (e.g. lock or use sanely!)
                inline const T* get() const
                {
                    if (ready())
                        return storage_t::getStorage();
                    return nullptr;
                }

                //! Utility to write less code, WILL ASSERT IF IT FAILS! So don't use on futures that might be cancelled or fail.
                inline T copy() const
                {
                    const bool success = wait();
                    assert(success);
                    return *get();
                }

                //! NOTE: Deliberately named `...acquire` instead of `...lock` to make them incompatible with `std::unique_lock`
                // and other RAII locks as the blocking aquire can fail and that needs to be handled.
                class storage_lock_t final
                {
                        using state_enum = future_base_t::STATE;
                        future_t<T>* m_future;

                        //! constructor, arg is nullptr if locked
                        friend class future_t<T>;
                        inline storage_lock_t(future_t<T>* _future) : m_future(_future)
                        {
                            assert(!m_future || m_future->state.query()==state_enum::LOCKED);
                        }
                        //! as usual for "unique" things
                        inline storage_lock_t(const storage_lock_t&) = delete;
                        inline storage_lock_t& operator=(const storage_lock_t&) = delete;

                    public:
                        inline ~storage_lock_t()
                        {
                            if (m_future)
                                m_future->state.exchangeNotify<true>(state_enum::READY,state_enum::LOCKED);
                        }

                        //!
                        inline explicit operator bool()
                        {
                            return m_future;
                        }
                        inline bool operator!()
                        {
                            return !m_future;
                        }

                        //!
                        inline T* operator->() const
                        {
                            if (m_future)
                                return m_future->getStorage();
                            return nullptr;
                        }
                        template<typename U=T> requires (std::is_same_v<U,T> && !std::is_void_v<U>)
                        inline U& operator*() const {return *operator->();}

                        //! Can only be called once!
                        inline void discard()
                        {
                            assert(m_future);
                            m_future->destruct();
                            m_future->state.exchangeNotify<true>(state_enum::INITIAL,state_enum::LOCKED);
                            m_future = nullptr;
                        }
                        //! Can only be called once!
                        template<typename U=T> requires (std::is_same_v<U,T> && !std::is_void_v<U>)
                        inline void move_into(U& dst)
                        {
                            dst = std::move(operator*());
                            discard();
                        }
                };

                //! ANY THREAD [except WORKER]: If we're READY transition to LOCKED
                inline storage_lock_t try_acquire()
                {
                    auto expected = STATE::READY;
                    return storage_lock_t(state.tryTransition(STATE::LOCKED,expected) ? this:nullptr);
                }
                //! ANY THREAD [except WORKER]: Wait till we're either in READY and move us to LOCKED or bail on INITIAL
                // this accounts for being cancelled or consumed while waiting
                inline storage_lock_t acquire()
                {
                    return storage_lock_t(state.waitAbortableTransition(STATE::LOCKED,STATE::READY,STATE::INITIAL) ? this:nullptr);
                }

            protected:
                // to get access to the below
                friend class IAsyncQueueDispatcherBase;
                // construct the retval element 
                template <typename... Args>
                inline void construct(Args&&... args)
                {
                    storage_t::construct(std::forward<Args>(args)...);
                }
        };
        template<typename T>
        struct cancellable_future_t : public future_t<T>
        {
                using base_t = future_t<T>;
                std::atomic<request_base_t*> request = nullptr;

            public:
                inline ~cancellable_future_t()
                {
                    // try to cancel
                    cancel();
                }

                //! ANY THREAD [except WORKER]: Cancel pending request if we can, returns whether we actually managed to cancel
                inline bool cancel()
                {
                    auto expected = base_t::STATE::ASSOCIATED;
                    if (base_t::state.tryTransition(base_t::STATE::EXECUTING,expected))
                    {
                        // Since we're here we've managed to move from ASSOCIATED to fake "EXECUTING" this means that the Request is either:
                        // 1. RECORDING but after returning from `base_t::associate_request`
                        while (!request.load()) {}
                        // 2. PENDING
                        // 3. EXECUTING but before returning from `base_t::disassociate_request` cause there's a spinlock there
                        
                        request.exchange(nullptr)->cancel();

                        // after doing everything, we can mark ourselves as cleaned up
                        base_t::state.template exchangeNotify<false>(base_t::STATE::INITIAL, base_t::STATE::EXECUTING);
                        return true;
                    }
                    // we're here because either:
                    // - there was no work submitted
                    // - someone else cancelled
                    // - request is currently executing
                    // - request is ready
                    // - storage is locked/acquired
                    // sanity check (there's a tiny gap between transitioning to EXECUTING and disassociating request)
                    assert(expected==base_t::STATE::EXECUTING || request==nullptr);
                    return false;
                }

            private:
                inline void associate_request(request_base_t* req) override final
                {
                    base_t::associate_request(req);
                    request_base_t* prev = request.exchange(req);
                    // sanity check
                    assert(prev==nullptr);
                }
                inline bool disassociate_request() override final
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

    protected:
        template<typename T>
        static inline core::StorageTrivializer<T>* future_storage_cast(future_base_t* _future_base) {return static_cast<future_t<T>*>(_future_base);}
};

inline void IAsyncQueueDispatcherBase::request_base_t::finalize(future_base_t* fut)
{
    future = fut;
    future->associate_request(this);
    state.exchangeNotify<false>(STATE::PENDING,STATE::RECORDING);
}

inline IAsyncQueueDispatcherBase::future_base_t* IAsyncQueueDispatcherBase::request_base_t::wait()
{
    if (state.waitAbortableTransition(STATE::EXECUTING,STATE::PENDING,STATE::CANCELLED) && future->disassociate_request())
        return future;
    //assert(future->cancellable);
    future = nullptr;
    state.exchangeNotify<false>(STATE::INITIAL,STATE::CANCELLED);
    return nullptr;
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
* Required accessible public methods of class being CRTP parameter:
* 
* void init(internal_state_t*); // required only in case of custom internal state
*
* void exit(internal_state_t*); // optional, no `state` parameter in case of no internal state
* 
* // no `state` parameter in case of no internal state
* void process_request(future_base_t*, request_metadata_t&, internal_state_t&);
* 
* void background_work() // optional, does nothing if not provided
* 
* 
* The `lock()` will be called just before calling into `background_work()` and processing any requests via `process_request()`,
* `unlock()` will be called just after processing the request (if any).
*/
template<typename CRTP, typename request_metadata_t, uint32_t BufferSize=256u, typename InternalStateType=void>
class IAsyncQueueDispatcher : public IThreadHandler<CRTP,InternalStateType>, protected impl::IAsyncQueueDispatcherBase
{
        static_assert(BufferSize>0u, "BufferSize must not be 0!");
        static_assert(core::isPoT(BufferSize), "BufferSize must be power of two!");

    protected:
        using base_t = IThreadHandler<CRTP,InternalStateType>;
        friend base_t; // TODO: remove, some functions should just be protected

        struct request_t : public request_base_t
        {
            inline request_t() : request_base_t() {}

            request_metadata_t m_metadata = {};
        };

    private:
        constexpr static inline uint32_t MaxRequestCount = BufferSize;

        // maybe one day we'll abstract this into a lockless queue
        using atomic_counter_t = std::atomic_uint64_t;
        using counter_t = atomic_counter_t::value_type;

        request_t request_pool[MaxRequestCount];
        atomic_counter_t cb_begin = 0u;
        atomic_counter_t cb_end = 0u;

        static inline counter_t wrapAround(counter_t x)
        {
            constexpr counter_t Mask = static_cast<counter_t>(BufferSize) - static_cast<counter_t>(1);
            return x & Mask;
        }

    public:
        inline IAsyncQueueDispatcher() {}
        inline IAsyncQueueDispatcher(base_t::start_on_construction_t) : base_t(base_t::start_on_construction) {}

        using mutex_t = typename base_t::mutex_t;
        using lock_t = typename base_t::lock_t;
        using cvar_t = typename base_t::cvar_t;
        using internal_state_t = typename base_t::internal_state_t;

        template<typename T>
        using future_t = impl::IAsyncQueueDispatcherBase::future_t<T>;
        template<typename T>
        using cancellable_future_t = impl::IAsyncQueueDispatcherBase::cancellable_future_t<T>;

        //! Constructs a request with `args` via `CRTP::request_impl` on the circular buffer after there's enough space to accomodate it.
        //! Then it associates the request to a future passed in as the first argument.
        template<typename T, typename... Args>
        void request(future_t<T>* _future, Args&&... args)
        {
            // get next output index
            const auto virtualIx = cb_end++;
            // protect against overflow by waiting for the worker to catch up
            const auto safe_begin = virtualIx<MaxRequestCount ? static_cast<counter_t>(0) : (virtualIx-MaxRequestCount+1u);
            for (counter_t old_begin; (old_begin=cb_begin.load())<safe_begin; )
                cb_begin.wait(old_begin);

            // get actual storage index now
            const auto r_id = wrapAround(virtualIx);

            request_t& req = request_pool[r_id];
            req.start();
            req.m_metadata = request_metadata_t(std::forward<Args>(args)...);
            req.finalize(_future);

            {
                auto global_lk = base_t::createLock();
                // wake up queue thread (needs to happen under a lock to not miss a wakeup)
                base_t::m_cvar.notify_one();
            }
        }

    protected:
        inline ~IAsyncQueueDispatcher() {}
        inline void background_work() {}

    private:
        template<typename... Args>
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
                if (future_base_t* future=req.wait())
                {
                    // if the request supports cancelling and got cancelled, then `wait()` function may return false
                    static_cast<CRTP*>(this)->process_request(future,req.m_metadata,optional_internal_state...);
                    req.notify();
                }
                // wake the waiter up
                cb_begin++;
                // this does not need to happen under a lock, because its not a condvar
                cb_begin.notify_one();
            }
            lock.lock();
        }

        inline bool wakeupPredicate() const { return (cb_begin != cb_end); }
        inline bool continuePredicate() const { return (cb_begin != cb_end); }
};

}

#endif
