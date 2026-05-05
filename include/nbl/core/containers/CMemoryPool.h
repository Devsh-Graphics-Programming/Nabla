// Copyright (C) 2020-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_C_MEMORY_POOL_H_INCLUDED_
#define _NBL_CORE_C_MEMORY_POOL_H_INCLUDED_


#include "nbl/core/decl/compile_config.h"
#include "nbl/core/decl/BaseClasses.h"
#include "nbl/core/alloc/SimpleBlockBasedAllocator.h"

#include <type_traits>


namespace nbl::core
{

template<typename C>
concept MemoryPoolConfig = requires
{
//    {C::ThreadSafe} -> std::same_as<bool>; // TODO: how to do it
    // Multi-thread safe address allocators shouldn't be used in the MemoryPoolConfigs, because the mem pool needs to sync more stuff
    typename C::AddressAllocator; // TODO: check its an Address Allocator, can we check its not MT?
    typename C::HandleValue;
};

template<MemoryPoolConfig Config>
class CMemoryPool final : public Uncopyable
{
        using block_allocator_st_type = SimpleBlockBasedAllocatorST<typename Config::AddressAllocator,typename Config::HandleValue>;
    public:
        using addr_allocator_type = Config::AddressAllocator;
        using size_type = typename core::address_allocator_traits<addr_allocator_type>::size_type;
		template<typename T>
		using typed_pointer_type = block_allocator_st_type::template typed_pointer_type<T>;

        using block_allocator_type = std::conditional_t<Config::ThreadSafe,SimpleBlockBasedAllocatorMT<block_allocator_st_type,std::recursive_mutex>,block_allocator_st_type>;

        using creation_params_type = block_allocator_st_type::SCreationParams;
        inline CMemoryPool(creation_params_type&& params) : m_block_alctr(std::move(params)) {}
        
		//
		template<typename T> requires (!std::is_const_v<T>)
		inline T* deref(typed_pointer_type<T> p)
		{
			return m_block_alctr.deref<T>(p);
		}
		template<typename T>
		inline const T* deref(typed_pointer_type<T> p) const
		{
			return m_block_alctr.deref<const T>(p);
		}
    
        //
        inline typed_pointer_type<void> allocate(const size_type s, const size_type a)
        {
            return m_block_alctr.allocate(s,a);
        }
        inline void deallocate(typed_pointer_type<void> _ptr, const size_type s)
        {
            m_block_alctr.deallocate(_ptr,s);
        }

        //
        template <typename T, typename... FuncArgs> requires (!std::is_array_v<T>) // for now until we have a test
        inline typed_pointer_type<T> emplace_n(const size_type n, FuncArgs&&... args)
        {
            size_type s = static_cast<size_type>(sizeof(T)*n);
            constexpr size_type a = alignof(T);
            const auto retval = block_allocator_st_type::template _reinterpret_cast<T>(allocate(s,a));
            if constexpr (!std::is_trivial_v<T>)
            if (retval)
            {
                T* ptr = deref<T>(retval);
                if constexpr (sizeof...(FuncArgs)!=0u)
                {
                    for (uint32_t i=0u; i<n; ++i)
                        std::construct_at(ptr+i,std::forward<FuncArgs>(args)...);
                }
                else
                    std::uninitialized_default_construct_n(ptr,n);
            }
            return retval;
        }
        template <typename T, typename... FuncArgs>
        inline typed_pointer_type<T> emplace(FuncArgs&&... args)
        {
            return emplace_n<T,FuncArgs...>(1u,std::forward<FuncArgs>(args)...);
        }

        // You must know the original type, we don't keep track of original size
        template <typename T> requires (!std::is_array_v<T>) // for now until we have a test
        inline void _delete(const typed_pointer_type<T> h, const size_type n=1)
        {
            if constexpr (!std::is_trivially_destructible_v<T>)
                std::destroy_n(deref<T>(h),n);
            deallocate(h,static_cast<size_type>(sizeof(T)*n));
        }

        //! Extra == Use WITH EXTREME CAUTION
        template<bool ThreadSafe=Config::ThreadSafe> requires (ThreadSafe && ThreadSafe==Config::ThreadSafe)
        inline std::recursive_mutex& get_lock() noexcept
        {
            return m_block_alctr.get_lock();
        }

    private:
        block_allocator_type m_block_alctr;
};

}
#endif
