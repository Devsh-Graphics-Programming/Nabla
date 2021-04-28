#ifndef __NBL_S_READ_WRITE_SPIN_LOCK_H_INCLUDED__
#define __NBL_S_READ_WRITE_SPIN_LOCK_H_INCLUDED__

#include <atomic>
#include <thread>

namespace nbl {
namespace system
{

namespace impl
{

    struct SReadWriteSpinLockBase
    {
        static inline constexpr uint32_t LockWriteVal = (1u << 31);
    };

}

struct SReadWriteSpinLock : protected impl::SReadWriteSpinLockBase
{
    SReadWriteSpinLock() : m_lock(0u) {}

    void lock_read(std::memory_order order = std::memory_order_seq_cst)
    {
        if (m_lock.fetch_add(1u, order) > (LockWriteVal-1u))
        {
            while (m_lock.load(order) >= LockWriteVal)
                std::this_thread::yield();
        }
    }

    void unlock_read(std::memory_order order = std::memory_order_seq_cst)
    {
        m_lock.fetch_sub(1u, order);
    }

    void lock_write(std::memory_order order = std::memory_order_seq_cst)
    {
        uint32_t expected = 0u;
        while (!m_lock.compare_exchange_strong(expected, LockWriteVal, order))
        {
            expected = 0u;
            std::this_thread::yield();
        }
    }

    void unlock_write(std::memory_order order = std::memory_order_seq_cst)
    {
        m_lock.fetch_sub(LockWriteVal, order);
    }

protected:
    std::atomic_uint32_t m_lock;
};


struct adopt_lock_t {};
constexpr inline adopt_lock_t adopt_lock;

template <std::memory_order Order = std::memory_order_seq_cst>
class read_lock_guard
{
    read_lock_guard(SReadWriteSpinLock& lk, adopt_lock_t) : m_lock(lk) {}
    explicit read_lock_guard(SReadWriteSpinLock& lk) : read_lock_guard(lk, adopt_lock)
    {
        m_lock.lock_read(Order);
    }

    ~read_lock_guard()
    {
        m_lock.unlock_read(Order);
    }

private:
    SReadWriteSpinLock& m_lock;
};

template <std::memory_order Order = std::memory_order_seq_cst>
class write_lock_guard
{
    write_lock_guard(SReadWriteSpinLock& lk, adopt_lock_t) : m_lock(lk) {}
    explicit write_lock_guard(SReadWriteSpinLock& lk) : write_lock_guard(lk, adopt_lock)
    {
        m_lock.lock_write(Order);
    }

    ~write_lock_guard()
    {
        m_lock.unlock_write(Order);
    }

private:
    SReadWriteSpinLock& m_lock;
};

}
}

#endif
