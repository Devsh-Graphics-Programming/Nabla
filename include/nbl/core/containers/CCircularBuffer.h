#ifndef __NBL_C_CIRCULAR_BUFFER_H_INCLUDED__
#define __NBL_C_CIRCULAR_BUFFER_H_INCLUDED__

#include <atomic>
#include <thread>

#include "nbl/core/compile_config.h"
#include "nbl/core/memory/memory.h"
#include "nbl/core/math/intutil.h"

namespace nbl {
namespace core
{

namespace impl
{

template <typename T>
class CRuntimeSizedCircularBufferBase
{
protected:
    static constexpr inline auto Alignment = alignof(T);

    explicit CRuntimeSizedCircularBufferBase(size_t cap) : m_mem(nullptr), m_capacity(cap)
    {
        assert(core::isPoT(cap));

        const size_t s = sizeof(T) * m_capacity;
        m_mem = _NBL_ALIGNED_MALLOC(s, Alignment);
    }

    T* getStorage()
    {
        return reinterpret_cast<T*>(m_mem);
    }

    const T* getStorage() const
    {
        return reinterpret_cast<const T*>(m_mem);
    }

public:
    using type = T;

    ~CRuntimeSizedCircularBufferBase()
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
    size_t m_capacity;
};


template <typename T, size_t S>
class CCompileTimeSizedCircularBufferBase
{
    static_assert(core::isPoT(S), "Circular buffer capacity must be PoT!");

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

public:
    using type = T;

    CCompileTimeSizedCircularBufferBase() = default;

    constexpr size_t capacity() const
    {
        return S;
    }

private:
    static constexpr inline auto MemSize = sizeof(T) * S;

    alignas(Alignment) uint8_t m_mem[MemSize];
};


template <typename B>
class CCircularBufferBase : public B
{
    using this_type = CCircularBufferBase<B>;
    using base_t = B;
    using type = typename base_t::type;

    using atomic_counter_t = std::atomic_uint64_t;
    using counter_t = atomic_counter_t::value_type;

    atomic_counter_t m_cb_begin;
    atomic_counter_t m_cb_end;

protected:
    constexpr static inline uint8_t UninitializedByte = 0xccU;
    void markStorageAsUninitialized()
    {
        auto* storage = base_t::getStorage();
        const size_t s = sizeof(type) * base_t::capacity();
        memset(storage, UninitializedByte, s);
    }
    bool isInitialized(const type* x) const
    {
        auto* xx = reinterpret_cast<const uint8_t*>(x);
        for (size_t i = 0ull; i < sizeof(type); ++i)
            if (xx[i] != UninitializedByte)
                return true;
        return false;
    }
    bool isInitialized(uint32_t ix) const
    {
        const type* x = getStorage() + ix;
        return isInitialized(x);
    }

private:
    static inline counter_t wrapAround(counter_t x)
    {
        const counter_t mask = static_cast<counter_t>(base_t::capacity()) - static_cast<counter_t>(1);
        return x & mask;
    }

    template <typename... Args>
    void push_back_impl(Args&&... args)
    {
        auto virtualIx = m_cb_end++;
        auto safe_begin = virtualIx<base_t::capacity() ? static_cast<counter_t>(0) : (virtualIx-base_t::capacity()+1u);

        for (counter_t old_begin; (old_begin = m_cb_begin.load()) < safe_begin; )
        {
#if __cplusplus >= 202002L
            m_cb_begin.wait(old_begin);
#else
            std::this_thread::yield();
#endif
        }

        const auto ix = wrapAround(virtualIx);

        type* storage = base_t::getStorage() + ix;
        if (isInitialized(storage))
            storage->~type();
        new (storage) type(std::forward<Args>(args)...);
    }

public:
    using base_t::base_t;

    friend class iterator
    {
        using difference_type = ptrdiff_t;

        friend this_type;
        this_type* m_owner;
        counter_t m_ix;

        explicit iterator(this_type* owner, counter_t ix) : m_owner(owner), m_ix(ix) {}

        counter_t getIx() const
        {
            return this_type::wrapAround(m_ix);
        }

    public:
        iterator operator+(difference_type n) const
        {
            return iterator(m_ix + static_cast<counter_t>(n));
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
            auto d = getIx() - rhs.getIx();
            return static_cast<difference_type>(d);
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
        iterator operator--()
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

    void push_back(const type& a)
    {
        push_back_impl(a);
    }
    void push_back(type&& a)
    {
        push_back_impl(std::move(a));
    }
    template <typename... Args>
    void emplace_back(Args&&... args)
    {
        push_back_impl(std::forward<Args>(args)...);
    }

    type& pop_front()
    {
        counter_t ix = m_cb_begin++;
#if __cplusplus >= 202002L
        m_cb_begin.notify_one();
#endif
        ix = wrapAround(ix);

        return base_t::getStorage()[ix];
    }

    // Using iterators is not thread-safe!
    // It is encouraged to have circular buffer externally synchronized while looping over it with begin(),end() etc.
    iterator begin()
    {
        counter_t b = m_cb_begin.load();
        return iterator(b);
    }
    iterator end()
    {
        counter_t e = m_cb_end.load() + 1u;
        return iterator(e);
    }

    size_t size() const
    {
        return m_cb_end.load() - m_cb_begin.load();
    }
};

}

template <typename T, size_t S>
class CCompileTimeSizedCircularBuffer : public impl::CCircularBufferBase<impl::CCompileTimeSizedCircularBufferBase<T, S>>
{
    using base_t = impl::CCircularBufferBase<impl::CCompileTimeSizedCircularBufferBase<T, S>>;

public:
    CCompileTimeSizedCircularBuffer()
    {
        base_t::markStorageAsUninitialized();
    }
};

template <typename T>
class CRunTimeSizedCircularBuffer : public impl::CCircularBufferBase<impl::CRunTimeSizedCircularBufferBase<T>>
{
    using base_t = impl::CCircularBufferBase<impl::CRunTimeSizedCircularBufferBase<T>>;

public:
    explicit CRunTimeSizedCircularBuffer(size_t cap) : base_t(cap) 
    {
        base_t::markStorageAsUninitialized();
    }
};

}
}

#endif
