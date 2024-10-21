#ifndef __NBL_C_CIRCULAR_BUFFER_H_INCLUDED__
#define __NBL_C_CIRCULAR_BUFFER_H_INCLUDED__

#include "nbl/core/decl/Types.h"
#include "nbl/core/memory/memory.h"
#include "nbl/core/math/intutil.h"

#include <atomic>
#include <thread>
#include <iterator>

namespace nbl::core
{

namespace impl
{

class CCircularBufferCommonBase
{
protected:
    // Instead of atomics for flags, we could use more memory (1 byte per flag)
    // In case of 1 bit per flag, atomic is a must
    using atomic_alive_flags_block_t = std::atomic_uint64_t;
    static inline constexpr auto bits_per_flags_block = 8ull * sizeof(atomic_alive_flags_block_t::value_type);

    constexpr static size_t numberOfFlagBlocksNeeded(size_t cap)
    {
        auto n_blocks = (cap + bits_per_flags_block - 1ull) / bits_per_flags_block;
        return n_blocks;
    }
};

template <typename T>
class CConstantRuntimeSizedCircularBufferBase : public CCircularBufferCommonBase
{
protected:
    static constexpr inline auto Alignment = alignof(T);

    explicit CConstantRuntimeSizedCircularBufferBase(size_t cap) : m_mem(nullptr), m_capacity(cap)
    {
        assert(core::isPoT(cap));

        const size_t s = sizeof(T) * m_capacity;
        m_mem = _NBL_ALIGNED_MALLOC(s, Alignment);
        memset(m_mem, 0, s);

        auto n_blocks = numberOfFlagBlocksNeeded(cap);
        m_flags = std::make_unique<atomic_alive_flags_block_t[]>(n_blocks);
        for (size_t i = 0u; i < n_blocks; ++i)
            std::atomic_init(m_flags.get() + i, 0ul);
    }

    T* getStorage()
    {
        return reinterpret_cast<T*>(m_mem);
    }

    const T* getStorage() const
    {
        return reinterpret_cast<const T*>(m_mem);
    }

    atomic_alive_flags_block_t* getAliveFlagsStorage()
    {
        return m_flags.get();
    }

    const atomic_alive_flags_block_t* getAliveFlagsStorage() const
    {
        return m_flags.get();
    }

public:
    using type = T;

    ~CConstantRuntimeSizedCircularBufferBase()
    {
        if (m_mem)
        {
            _NBL_ALIGNED_FREE(m_mem);
        }
    }

    size_t capacity() const
    {
        return m_capacity;
    }

private:
    void* m_mem;
    std::unique_ptr<atomic_alive_flags_block_t[]> m_flags;
    size_t m_capacity;
};


template <typename T, size_t S>
class CCompileTimeSizedCircularBufferBase : public CCircularBufferCommonBase
{
    static_assert(core::isPoT(S), "Circular buffer capacity must be PoT!");

    static constexpr inline auto StorageSize = sizeof(T) * S;

protected:
    static constexpr inline auto Alignment = alignof(T);

    T* getStorage()
    {
        return reinterpret_cast<T*>(m_mem);
    }

    const T* getStorage() const
    {
        return reinterpret_cast<const T*>(m_mem);
    }

    atomic_alive_flags_block_t* getAliveFlagsStorage()
    {
        return m_flags;
    }

    const atomic_alive_flags_block_t* getAliveFlagsStorage() const
    {
        return m_flags;
    }

public:
    using type = T;

    CCompileTimeSizedCircularBufferBase()
    {
        for (auto& a : m_flags)
            std::atomic_init(&a, 0);
    }

    constexpr size_t capacity() const
    {
        return S;
    }

private:
    alignas(Alignment) uint8_t m_mem[StorageSize] {};
    atomic_alive_flags_block_t m_flags[CCircularBufferCommonBase::numberOfFlagBlocksNeeded(S)];
};

// Do not use with AllowOverflows and non PoD data types concurrently, you can get unordered and non-atomic construction and destruction of elements
// TODO: Reimplement with per-element C++20 atomic waits and a ticket lock (for ordered overwrites)
// https://cdn.discordapp.com/attachments/593903264987349057/872793258042986496/100543_449723386_Bryce_Adelstein_Lelbach_The_C20_synchronization_library.pdf
template <typename Base, bool AllowOverflows = true>
class CCircularBufferBase : public Base
{
    using this_type = CCircularBufferBase<Base>;
    using base_t = Base;
    using type = typename base_t::type;
    static_assert(!AllowOverflows || std::is_trivially_destructible_v<type>);

    using atomic_counter_t = std::atomic_uint64_t;
    using counter_t = atomic_counter_t::value_type;

    atomic_counter_t m_cb_begin = 0;
    atomic_counter_t m_cb_end = 0;

protected:
    bool isAlive(uint32_t ix) const
    {
        const auto* flags = base_t::getAliveFlagsStorage();
        const auto block_n = ix / base_t::bits_per_flags_block;
        const auto block = flags[block_n].load();
        const auto local_ix = ix & (base_t::bits_per_flags_block - 1u);

        return (block >> local_ix) & static_cast<typename base_t::atomic_alive_flags_block_t::value_type>(1);
    }
    void flipAliveFlag(uint32_t ix)
    {
        const auto block_n = ix / base_t::bits_per_flags_block;
        const auto local_ix = ix & (base_t::bits_per_flags_block - 1u);

        auto xormask = static_cast<typename base_t::atomic_alive_flags_block_t::value_type>(1) << local_ix;

        auto* flags = base_t::getAliveFlagsStorage();
        flags[block_n].fetch_xor(xormask);
    }

private:
    counter_t wrapAround(counter_t x)
    {
        const counter_t mask = static_cast<counter_t>(base_t::capacity()) - static_cast<counter_t>(1);
        return x & mask;
    }

    template <typename... Args>
    type& push_back_impl(Args&&... args)
    {
        auto virtualIx = m_cb_end++;

        if constexpr (!AllowOverflows)
        {
            auto safe_begin = virtualIx < base_t::capacity() ? static_cast<counter_t>(0) : (virtualIx - base_t::capacity() + 1u);
            for (counter_t old_begin; (old_begin = m_cb_begin.load()) < safe_begin; )
                m_cb_begin.wait(old_begin);
        }

        const auto ix = wrapAround(virtualIx);

        type* storage = base_t::getStorage() + ix;
        const bool was_alive = isAlive(ix);
        if (was_alive)
            storage->~type();
        new (storage) type(std::forward<Args>(args)...);
        if (!was_alive)
            flipAliveFlag(ix);

        return *storage;
    }

public:
    using base_t::base_t;

    class iterator
    {
        friend this_type;
        this_type* m_owner;
        counter_t m_ix;

        explicit iterator(this_type* owner, counter_t ix) : m_owner(owner), m_ix(ix) {}

        counter_t getIx() const
        {
            return m_owner->wrapAround(m_ix);
        }

    public:
        using value_type = type;
        using pointer = type*;
        using reference = type&;
        using difference_type = ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;

        iterator operator+(difference_type n) const
        {
            return iterator(m_owner,m_ix+static_cast<counter_t>(n));
        }
        iterator& operator+=(difference_type n)
        {
            m_ix += static_cast<counter_t>(n);
            return *this;
        }
        iterator operator-(difference_type n) const
        {
            return operator+(-n);
        }
        iterator& operator-=(difference_type n)
        {
            return operator+=(-n);
        }
        difference_type operator-(const iterator& rhs) const
        {
            return static_cast<difference_type>(m_ix-rhs.m_ix);
        }
        iterator& operator++()
        {
            return operator+=(1);
        }
        iterator operator++(int)
        {
            auto cp = *this;
            ++(*this);
            return cp;
        }
        iterator& operator--()
        {
            return operator-=(1);
        }
        iterator operator--(int)
        {
            auto cp = *this;
            --(*this);
            return cp;
        }

        type& operator*()
        {
            return m_owner->getStorage()[getIx()];
        }
        type& operator[](difference_type n)
        {
            return *((*this) + n);
        }

        bool operator==(const iterator& rhs) const
        {
            return m_ix == rhs.m_ix;
        }
        bool operator!=(const iterator& rhs) const
        {
            return m_ix != rhs.m_ix;
        }
        bool operator<(const iterator& rhs) const
        {
            return this->operator-(rhs) > 0;
        }
        bool operator>(const iterator& rhs) const
        {
            return rhs < (*this);
        }
        bool operator>=(const iterator& rhs) const
        {
            return !(this->operator<(rhs));
        }
        bool operator<=(const iterator& rhs) const
        {
            return !(this->operator>(rhs));
        }
    };
    friend class iterator;

    void push_back(const type& a)
    {
        push_back_impl(a);
    }
    void push_back(type&& a)
    {
        push_back_impl(std::move(a));
    }
    template <typename... Args>
    type& emplace_back(Args&&... args)
    {
        return push_back_impl(std::forward<Args>(args)...);
    }

    type pop_front()
    {
        counter_t ix = m_cb_begin.load();
        ix = wrapAround(ix);
        #ifdef _NBL_DEBUG
            if constexpr (!AllowOverflows)
            {
                bool alive = isAlive(ix);
                assert(alive);
            }
        #endif

        type* storage = base_t::getStorage() + ix;
        auto cp = std::move(*storage);
        flipAliveFlag(ix);
        storage->~type();

        ++m_cb_begin;
        if constexpr (!AllowOverflows)
        {
            m_cb_begin.notify_one();
        }

        return cp;
    }

    // Using iterators is not thread-safe!
    // It is encouraged to have circular buffer externally synchronized while looping over it with begin(),end() etc.
    iterator begin()
    {
        counter_t b = m_cb_begin.load();
        return iterator(this,b);
    }
    iterator end()
    {
        counter_t e = m_cb_end.load();
        return iterator(this,e);
    }

    size_t size() const
    {
        return m_cb_end.load() - m_cb_begin.load();
    }
};

}

template <typename T, size_t S, bool AllowOverflows = true>
class CCompileTimeSizedCircularBuffer : public impl::CCircularBufferBase<impl::CCompileTimeSizedCircularBufferBase<T, S>, AllowOverflows>
{
    using base_t = impl::CCircularBufferBase<impl::CCompileTimeSizedCircularBufferBase<T, S>>;

public:
    CCompileTimeSizedCircularBuffer() = default;
};

template <typename T, bool AllowOverflows = true>
class CConstantRuntimeSizedCircularBuffer : public impl::CCircularBufferBase<impl::CConstantRuntimeSizedCircularBufferBase<T>, AllowOverflows>
{
    using base_t = impl::CCircularBufferBase<impl::CConstantRuntimeSizedCircularBufferBase<T>>;

public:
    explicit CConstantRuntimeSizedCircularBuffer(size_t cap) : base_t(cap)
    {

    }
};

}

#endif
