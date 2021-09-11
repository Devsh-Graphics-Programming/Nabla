#ifndef __NBL_C_MEMORY_POOL_H_INCLUDED__
#define __NBL_C_MEMORY_POOL_H_INCLUDED__

#include "nbl/core/decl/compile_config.h"
#include "nbl/core/alloc/SimpleBlockBasedAllocator.h"
#include "nbl/core/decl/BaseClasses.h"

#include <memory>
#include <type_traits>

namespace nbl::core
{

template <class AddressAllocator, template<class> class DataAllocator, bool isThreadSafe, typename... Args>
class CMemoryPool : public Uncopyable
{
public:
    using addr_allocator_type = AddressAllocator;
    using allocator_type = std::conditional<isThreadSafe,
        SimpleBlockBasedAllocatorMT<AddressAllocator,DataAllocator, std::recursive_mutex, Args...>,
        SimpleBlockBasedAllocatorST<AddressAllocator,DataAllocator, Args...>>::type;
    using size_type = typename core::address_allocator_traits<addr_allocator_type>::size_type;
    using addr_type = size_type;

    CMemoryPool(size_type _blockSize, size_type _minBlockCount, size_type _maxBlockCount, Args... args) : // intentionally no && here, i dont wont to do here anything like reference collapsing, `Args` come from class template
        m_alctr(_blockSize,_minBlockCount,_maxBlockCount,std::forward<Args>(args)...)
    {
    }
    
    void* allocate(size_type s, size_type a)
    {
        return m_alctr.allocate(s, a);
    }
    void deallocate(void* _ptr, size_type s)
    {
        m_alctr.deallocate(_ptr, s);
    }

    template <typename T, typename... FuncArgs>
    T* emplace_n(uint32_t n, FuncArgs&&... args)
    {
        size_type s = static_cast<size_type>(n) * sizeof(T);
        size_type a = alignof(T);
        void* ptr = allocate(s, a);
        if (!ptr)
            return nullptr;

        using traits_t = std::allocator_traits<DataAllocator<T>>;
        DataAllocator<T> data_alctr;
        if constexpr (sizeof...(FuncArgs)!=0u || !std::is_pod_v<T>)
        {
            for (uint32_t i = 0u; i < n; ++i)
                traits_t::construct(data_alctr, reinterpret_cast<T*>(ptr) + i, std::forward<FuncArgs>(args)...);
        }
        return reinterpret_cast<T*>(ptr);
    }
    template <typename T, typename... FuncArgs>
    T* emplace(FuncArgs&&... args)
    {
        return emplace_n<T,FuncArgs...>(1u, std::forward<FuncArgs>(args)...);
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
        deallocate(_ptr, sizeof(T)*n);
    }
    template <typename T>
    void free(void* ptr)
    {
        return free_n<T>(ptr, 1u);
    }

private:
    allocator_type m_alctr;
};

}

#endif
