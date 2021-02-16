#ifndef __NBL_I_THREAD_HANDLER_H_INCLUDED__
#define __NBL_I_THREAD_HANDLER_H_INCLUDED__

#include <mutex>
#include <condition_variable>
#include <thread>

namespace nbl {
namespace system
{

// Usage:
/*
* class MyThreadHandler : public IThreadHandler<SomeInternalStateType> { .... };
* 
* MyThreadHandler handler;
* std::thread thread(&MyThreadHandler::thread, &handler);
* //...
* //... communicate with the thread using your methods (see note at the end of this section)), thread will sleep until wakeupPredicate() returns true
* //...
* handler.terminate(thread);
* After this handler can be safely destroyed.
* Every method playing around with object's state shared with the thread must begin with line: `auto raii_handler = createRAIIDisptachHandler();`!
*/
template <typename InternalStateType = int>
class IThreadHandler
{
protected:
    using mutex_t = std::mutex;
    using cvar_t = std::condition_variable;
    using lock_t = std::unique_lock<mutex_t>;

    using internal_state_t = InternalStateType;

    struct raii_dispatch_handler_t
    {
        raii_dispatch_handler_t(lock_t&& _lk, cvar_t& _cv) : lk(std::move(lk)), cv(_cv) {}
        ~raii_dispatch_handler_t()
        {
            lk.unlock();
            cv.notify_one();
        }

    private:
        lock_t lk;
        cvar_t& cv;
    };

    inline lock_t createLock() { return lock_t{ m_mutex }; }
    inline raii_dispatch_handler_t createRAIIDispatchHandler() { return raii_dispatch_handler_t(createLock(), m_cvar); }

    virtual internal_state_t init() = 0;
    virtual bool wakeupPredicate() const { return m_quit; }
    virtual bool continuePredicate() const { return !m_quit; }

    // lock is locked at the beginning of this function and must be locked at the exit
    virtual void work(lock_t& lock, internal_state_t& state) = 0;

    virtual void exit(internal_state_t& state) {}

public:
    void thread()
    {
        auto state = init();

        auto lock = createLock();

        do {
            m_cvar.wait(lock, [this] { return this->wakeupPredicate(); });

            if (continuePredicate())
            {
                work(lock, state);
            }
        } while (!m_quit);

        exit(state);
    }

    void terminate(std::thread& th)
    {
        auto lock = createLock();
        m_quit = true;
        lock.unlock();
        m_cvar.notify_one();

        if (th.joinable())
            th.join();
    }

    virtual ~IThreadHandler() = default;

private:
    mutex_t m_mutex;
    cvar_t m_cvar;
    bool m_quit = false;
};

}
}


#endif