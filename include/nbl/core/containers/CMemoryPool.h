// Copyright (C) 2020-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_C_MEMORY_POOL_H_INCLUDED_
#define _NBL_CORE_C_MEMORY_POOL_H_INCLUDED_


#include "nbl/core/decl/compile_config.h"
#include "nbl/core/decl/BaseClasses.h"
#include "nbl/core/alloc/SimpleBlockBasedAllocator.h"

#include <memory>
#include <type_traits>


namespace nbl::core
{

template<typename C>
concept MemoryPoolConfig = requires
{
//    {C::ThreadSafe} -> std::same_as<bool>; // TODO: how to do it
    typename C::AddressAllocator; // TODO: check its an Address Allocator
};

template<MemoryPoolConfig Config>
class CMemoryPool final : public Uncopyable
{
        using block_allocator_st_type = SimpleBlockBasedAllocatorST<typename Config::AddressAllocator>;
    public:
        using addr_allocator_type = Config::AddressAllocator;
        using size_type = typename core::address_allocator_traits<addr_allocator_type>::size_type;

        using block_allocator_type = std::conditional_t<Config::ThreadSafe,SimpleBlockBasedAllocatorMT<block_allocator_st_type,std::recursive_mutex>,block_allocator_st_type>;
// TODO: not appropriate
//        using addr_type = size_type;

        inline CMemoryPool(block_allocator_st_type::SCreationParams&& params) : m_block_alctr(std::move(params)) {}
    
        //
        inline void* allocate(const size_type s, const size_type a)
        {
            return m_block_alctr.allocate(s,a);
        }
        inline void deallocate(void* _ptr, const size_type s)
        {
            m_block_alctr.deallocate(_ptr,s);
        }

        //
        template <typename T, typename... FuncArgs> requires (!std::is_array_v<T>) // for now until we have a test
        inline T* emplace_n(const uint32_t n, FuncArgs&&... args)
        {
            size_type s = static_cast<size_type>(n)*sizeof(T);
            size_type a = alignof(T);
            T* const ptr = std::launder(reinterpret_cast<T*>(allocate(s,a)));
            if (!ptr)
                return nullptr;

            if constexpr (!std::is_trivial_v<T>)
            {
                if constexpr (sizeof...(FuncArgs)!=0u)
                {
                    for (uint32_t i=0u; i<n; ++i)
                        std::construct_at(ptr+i,std::forward<FuncArgs>(args)...);
                }
                else
                    std::uninitialized_default_construct_n(ptr,n);
            }
            return ptr;
        }
        template <typename T, typename... FuncArgs>
        inline T* emplace(FuncArgs&&... args)
        {
            return emplace_n<T,FuncArgs...>(1u,std::forward<FuncArgs>(args)...);
        }

        // You must know the original type, we don't keep track of original size
        // TODO: this shouldn't be called `free` but `delete`
        template <typename T> requires (!std::is_array_v<T>) // for now until we have a test
        inline void free_n(void* _ptr, const uint32_t n)
        {
            T* ptr = reinterpret_cast<T*>(_ptr);
            if constexpr (!std::is_trivially_destructible_v<T>)
                std::destroy_n(ptr,n);
            deallocate(_ptr,sizeof(T)*n);
        }
        template <typename T>
        inline void free(void* ptr)
        {
            return free_n<T>(ptr,1u);
        }

    private:
        block_allocator_type m_block_alctr;
};

}
#endif
