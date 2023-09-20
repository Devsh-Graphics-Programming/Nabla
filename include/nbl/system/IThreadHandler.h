#ifndef __NBL_I_THREAD_HANDLER_H_INCLUDED__
#define __NBL_I_THREAD_HANDLER_H_INCLUDED__

#include <mutex>
#include <condition_variable>
#include <thread>
#include <string>

namespace nbl::system
{

#if 0 // C++20 concepts suck at checking for non-public method existence!
namespace impl
{

// TODO: finish these concepts with `work`,`init` and `exit`
template<typename T>
concept ThreadHandler = requires(const T& cstate)
{
    {cstate.continuePredicate()} -> std::same_as<bool>;
    {cstate.wakeupPredicate()} -> std::same_as<bool>;
    {cstate.work()} -> std::same_as<void>;
};

template<typename T>
concept StatefulThreadHandler

}
#endif

// Usage:
/*
* class MyThreadHandler : public IThreadHandler<MyThreadHandler,SomeInternalStateType> { .... };
* 
* MyThreadHandler handler;
* std::thread thread(&MyThreadHandler::thread, &handler);
* //...
* //... communicate with the thread using your methods (see note at the end of this section)), thread will sleep until wakeupPredicate() returns true
* //...
* handler.terminate(thread);
* After this handler can be safely destroyed.
* 
* Every method playing around with object's state shared with the thread must begin with line: `auto raii_handler = createRAIIDisptachHandler();`!
*/
template<typename CRTP, typename InternalStateType = void>
class IThreadHandler
{
    private:
        // TODO: factor out into `nbl/core/reflection/has_member_function.h`
    #define _NBL_IMPL_MEMBER_FUNC_PRESENCE_CHECKER(member_func_name)\
        class has_##member_func_name\
        {\
            using true_type = uint32_t;\
            using false_type = uint64_t;\
        \
            template <typename T>\
            static true_type& test(decltype(&T::member_func_name));\
            template <typename T>\
            static false_type& test(...);\
        \
        public:\
            static inline constexpr bool value = (sizeof(test<CRTP>(0)) == sizeof(true_type));\
        };

        _NBL_IMPL_MEMBER_FUNC_PRESENCE_CHECKER(init)
        _NBL_IMPL_MEMBER_FUNC_PRESENCE_CHECKER(exit)

    #undef _NBL_IMPL_MEMBER_FUNC_PRESENCE_CHECKER

    protected:
        using mutex_t = std::mutex;
        using cvar_t = std::condition_variable;
        using lock_t = std::unique_lock<mutex_t>;

        static inline constexpr bool has_internal_state = !std::is_void_v<InternalStateType>;
        using internal_state_t = std::conditional_t<has_internal_state, InternalStateType, int>;

        struct raii_dispatch_handler_t
        {
                raii_dispatch_handler_t(mutex_t& _mtx, cvar_t& _cv) : lk(_mtx), cv(_cv) {}
                ~raii_dispatch_handler_t()
                {
                    cv.notify_one();
                    // raii-style unlock happens in destructor of `lk` after notification
                }

            private:
                lock_t lk;
                cvar_t& cv;
        };

        inline lock_t createLock() { return lock_t(m_mutex); }
        inline lock_t tryCreateLock() { return lock_t(m_mutex,std::try_to_lock); }
        inline raii_dispatch_handler_t createRAIIDispatchHandler() { return raii_dispatch_handler_t(m_mutex, m_cvar); }

        // Required accessible methods of class being CRTP parameter:

        //void init(internal_state_t*); // required only in case of custom internal state, optional otherwise. Parameterless in case of no internal state
        //bool wakeupPredicate() const;
        //bool continuePredicate() const;

        // no `state` parameter in case of no internal state
        // lock is locked at the beginning of this function and must be locked at the exit
        //void work(lock_t& lock, internal_state_t& state);

        // lock is locked at the beginning of this function and must be locked at the exit
        //void exit(internal_state_t* state); // optional, no `state` parameter in case of no internal state

    private:
        internal_state_t* getInternalStatePtr() { return reinterpret_cast<internal_state_t*>(m_internal_state_storage); }

        inline void init_impl()
        {
            internal_state_t* state_ptr = getInternalStatePtr();

            if constexpr (has_internal_state)
            {
                static_cast<CRTP*>(this)->init(state_ptr);
            }
            else if (has_init::value)
            {
                static_cast<CRTP*>(this)->init();
            }
            m_initComplete.test_and_set();
            m_initComplete.notify_one();
        }

        void terminate()
        {
            {
                auto lockAndSignal = createRAIIDispatchHandler();
                m_quit = true;
            }

            if (m_thread.joinable())
                m_thread.join();
        }

    public:
        struct start_on_construction_t {};
        constexpr inline static start_on_construction_t start_on_construction {};

        IThreadHandler() : m_thread() {}
        IThreadHandler(start_on_construction_t) : m_thread(&IThreadHandler<CRTP,InternalStateType>::thread,this) {}

        //! Has no effect if thread is already running
        bool start()
        {
            if (m_thread.get_id() == std::thread::id())
            {
                m_thread = std::thread(&IThreadHandler<CRTP,InternalStateType>::thread,this);
                return true;
            }
            return false;
        }

        void waitForInitComplete()
        {
            m_initComplete.wait(false);
        }

        ~IThreadHandler()
        {
            terminate();
        }

    protected:
        void thread()
        {
            CRTP* this_ = static_cast<CRTP*>(this);

            init_impl();
            internal_state_t* state_ptr = getInternalStatePtr();

            auto lock = createLock();

            do {
                m_cvar.wait(lock, [this,this_] { return this_->wakeupPredicate() || this->m_quit; });

                if (this_->continuePredicate() && !m_quit)
                {
                    if constexpr (has_internal_state)
                    {
                        internal_state_t& internal_state = state_ptr[0];
                        this_->work(lock, internal_state);
                    }
                    else
                    {
                        this_->work(lock);
                    }
                }
            } while (!m_quit);

            if constexpr (has_exit::value)
            {
                if constexpr (has_internal_state)
                {
                    this_->exit(state_ptr);
                }
                else
                {
                    this_->exit();
                }
            }
        }

    protected:
        alignas(internal_state_t) uint8_t m_internal_state_storage[sizeof(internal_state_t)];

        mutex_t m_mutex;
        cvar_t m_cvar;
        std::atomic_flag m_initComplete; // begins in false state, per C++11 spec
        bool m_quit = false; // TODO: make this an atomic_flag

        // Must be last member!
        std::thread m_thread;
};

}


#endif