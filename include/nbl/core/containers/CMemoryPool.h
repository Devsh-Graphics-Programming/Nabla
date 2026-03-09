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
    typename C::AddressAllocator; // TODO: check its an Address Allocator
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
		struct typed_pointer
		{
			private:
				using __t = typename block_allocator_st_type::template typed_pointer<T>;
			public:
				using type = typename __t::type;
		};

        using block_allocator_type = std::conditional_t<Config::ThreadSafe,SimpleBlockBasedAllocatorMT<block_allocator_st_type,std::recursive_mutex>,block_allocator_st_type>;

        using creation_params_type = block_allocator_st_type::SCreationParams;
        inline CMemoryPool(creation_params_type&& params) : m_block_alctr(std::move(params)) {}
        
		//
		template<typename T> requires (!std::is_const_v<T>)
		inline T* deref(typename typed_pointer<T>::type p)
		{
			return m_block_alctr.deref<T>(p);
		}
		template<typename T> requires std::is_const_v<T>
		inline T* deref(typename typed_pointer<T>::type p) const
		{
			return m_block_alctr.deref<T>(p);
		}
    
        //
        inline typename typed_pointer<void>::type allocate(const size_type s, const size_type a)
        {
            return m_block_alctr.allocate(s,a);
        }
        inline void deallocate(typename typed_pointer<void>::type _ptr, const size_type s)
        {
            m_block_alctr.deallocate(_ptr,s);
        }

        //
        template <typename T, typename... FuncArgs> requires (!std::is_array_v<T>) // for now until we have a test
        inline typename typed_pointer<T>::type emplace_n(const uint32_t n, FuncArgs&&... args)
        {
            size_type s = static_cast<size_type>(n)*sizeof(T);
            size_type a = alignof(T);
            typename typed_pointer<T>::type ptr = std::launder(reinterpret_cast<T*>(allocate(s,a)));
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
        inline typename typed_pointer<T>::type emplace(FuncArgs&&... args)
        {
            return emplace_n<T,FuncArgs...>(1u,std::forward<FuncArgs>(args)...);
        }

        // You must know the original type, we don't keep track of original size
        template <typename T> requires (!std::is_array_v<T>) // for now until we have a test
        inline void _delete(const typename typed_pointer<T>::type h, const size_type n=1)
        {
            if constexpr (!std::is_trivially_destructible_v<T>)
                std::destroy_n(h,n);
            deallocate(h,sizeof(T)*n);
        }

    private:
        block_allocator_type m_block_alctr;
};

}
#endif
