#ifndef __NBL_S_READ_WRITE_SPIN_LOCK_H_INCLUDED__
#define __NBL_S_READ_WRITE_SPIN_LOCK_H_INCLUDED__

#include <atomic>
#include <thread>
#include <mutex> // for std::adopt_lock_t

namespace nbl {
namespace system
{

namespace impl
{

    class SReadWriteSpinLockBase
    {
    public:
        static inline constexpr uint32_t LockWriteVal = (1u << 31);

        // TODO atomic_unsigned_lock_free? since C++20
        std::atomic_uint32_t m_lock = 0u;
    };

}

template <std::memory_order>
class read_lock_guard;
template <std::memory_order>
class write_lock_guard;

class SReadWriteSpinLock : protected impl::SReadWriteSpinLockBase
{
    void lock_write_impl(const uint32_t _expected, std::memory_order order)
    {
        uint32_t expected = _expected;
        while (!m_lock.compare_exchange_strong(expected, LockWriteVal, order))
        {
            expected = _expected;
            std::this_thread::yield();
        }
    }

public:
    template <std::memory_order>
    friend class read_lock_guard;
    template <std::memory_order>
    friend class write_lock_guard;

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
        lock_write_impl(0u, order);
    }

    void unlock_write(std::memory_order order = std::memory_order_seq_cst)
    {
        m_lock.fetch_sub(LockWriteVal, order);
    }
};

namespace impl
{
    class rw_lock_guard_base
    {
        rw_lock_guard_base() : m_lock(nullptr) {}

    public:
        rw_lock_guard_base& operator=(const rw_lock_guard_base&) = delete;
        rw_lock_guard_base(const rw_lock_guard_base&) = delete;

        rw_lock_guard_base& operator=(rw_lock_guard_base&& rhs)
        {
            std::swap(m_lock, rhs.m_lock);
            return *this;
        }
        rw_lock_guard_base(rw_lock_guard_base&& rhs) : rw_lock_guard_base()
        {
            operator=(std::move(rhs));
        }

    protected:
        rw_lock_guard_base(SReadWriteSpinLock& lk) : m_lock(&lk) {}

        SReadWriteSpinLock* m_lock;
    };
}

template <std::memory_order Order = std::memory_order_seq_cst>
class read_lock_guard : public impl::rw_lock_guard_base
{
    read_lock_guard() : m_lock(nullptr) {}

public:
    read_lock_guard(SReadWriteSpinLock& lk, std::adopt_lock_t) : impl::rw_lock_guard_base(lk) {}
    explicit read_lock_guard(SReadWriteSpinLock& lk) : read_lock_guard(lk, adopt_lock)
    {
        m_lock->lock_read(Order);
    }
    explicit read_lock_guard(write_lock_guard<Order>&& wl);

    ~read_lock_guard()
    {
        if (m_lock)
            m_lock->unlock_read(Order);
    }
};

template <std::memory_order Order = std::memory_order_seq_cst>
class write_lock_guard : public impl::rw_lock_guard_base
{
    write_lock_guard() : m_lock(nullptr) {}

public:
    write_lock_guard(SReadWriteSpinLock& lk, std::adopt_lock_t) : impl::rw_lock_guard_base(lk) {}
    explicit write_lock_guard(SReadWriteSpinLock& lk) : write_lock_guard(lk, adopt_lock)
    {
        m_lock->lock_write(Order);
    }
    explicit write_lock_guard(read_lock_guard<Order>&& rl);

    ~write_lock_guard()
    {
        if (m_lock)
            m_lock->unlock_write(Order);
    }
};

template <std::memory_order Order>
inline read_lock_guard<Order>::read_lock_guard(write_lock_guard<Order>&& wl) : impl::rw_lock_guard_base(std::move(wl))
{
    m_lock->m_lock.fetch_sub(LockWriteVal - 1u);
}

template <std::memory_order Order>
inline write_lock_guard<Order>::write_lock_guard(read_lock_guard<Order>&& rl) : impl::rw_lock_guard_base(std::move(rl))
{
    m_lock->lock_write_impl(1u, Order);
}

}
}

#endif
