#ifndef __NBL_C_MEMORY_POOL_H_INCLUDED__
#define __NBL_C_MEMORY_POOL_H_INCLUDED__

#include "nbl/core/decl/compile_config.h"
#include "nbl/core/alloc/SimpleBlockBasedAllocator.h"
#include "nbl/core/decl/BaseClasses.h"

#include <memory>

namespace nbl {
namespace core
{

template <class AddressAllocator, template<class> class DataAllocator>
class CMemoryPool : public Uncopyable
{
public:
    using addr_allocator_type = AddressAllocator;
    using allocator_type = SimpleBlockBasedAllocator<AddressAllocator, DataAllocator, uint32_t>;
    using size_type = typename core::address_allocator_traits<addr_allocator_type>::size_type;
    using addr_type = size_type;

    CMemoryPool(size_type _blockSize, size_type _maxBlockCount) :
        m_alctr(_blockSize, _maxBlockCount, 1u)
    {

    }

    template <typename T, typename... Args>
    T* emplace_n(uint32_t n, Args&&... args)
    {
        size_type s = static_cast<size_type>(n) * sizeof(T);
        size_type a = alignof(T);
        void* ptr = alloc_addr(s, a);
        if (!ptr)
            return nullptr;

        using traits_t = std::allocator_traits<DataAllocator<T>>;
        DataAllocator<T> data_alctr;
        if constexpr (sizeof...(Args)!=0u || !std::is_pod_v<T>)
        {
            for (uint32_t i = 0u; i < n; ++i)
                traits_t::construct(data_alctr, reinterpret_cast<T*>(ptr) + i, std::forward<Args>(args)...);
        }
        return reinterpret_cast<T*>(ptr);
    }
    template <typename T, typename... Args>
    T* emplace(Args&&... args)
    {
        return emplace_n(1u, std::forward<Args>(args)...);
    }
    template <typename T>
    void free_n(void* _ptr, uint32_t n)
    {
        using traits_t = std::allocator_traits<DataAllocator<T>>;
        DataAllocator<T> data_alctr;

        T* ptr = reinterpret_cast<T*>(_ptr);
        if constexpr (!std::is_trivially_destructible_v<T>)
        {
            for (uint32_t i = 0u; i < n; ++i)
                traits_t::destroy(data_alctr, ptr + i);
        }
        m_alctr.deallocate(_ptr, sizeof(T)*n);
    }
    template <typename T>
    void free(void* ptr)
    {
        return free_n(ptr, 1u);
    }

private:
    void* alloc_addr(size_type s, size_type a)
    {
        return m_alctr.allocate(s, a);
    }

    allocator_type m_alctr;
};

}
}

#endif
