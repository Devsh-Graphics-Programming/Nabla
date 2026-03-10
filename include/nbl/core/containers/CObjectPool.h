// Copyright (C) 2026-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_C_OBJECT_POOL_H_INCLUDED_
#define _NBL_CORE_C_OBJECT_POOL_H_INCLUDED_


#include "nbl/core/containers/CMemoryPool.h"


namespace nbl::core
{

//
class IObjectPoolBase : public Uncopyable
{
	public:
		// Base class for objects whose destructors MUST be ran
		class INonTrivial
		{
			protected:
				//
				friend class IObjectPoolBase;

				// to not be able to make the variable length objects on the stack
				virtual ~INonTrivial() = 0;
		};
		// to support variable length objects
		class IVariableSize : public INonTrivial
		{
			public:
				// its not even needed right now
//				virtual uint32_t getSize() const = 0;
		};
		
	protected:
		virtual ~IObjectPoolBase() = 0;
		
		// skip the lock, deallocation and record erase, returns total size
		template<typename size_type>
		inline void destroy(INonTrivial* p, const size_type n, const size_type stride)
		{
			for (size_type i=0; i<n; i++)
			{
				// can't use `std::destroy_at<T>(ptr);` because of destructor being non-public
				static_cast<INonTrivial*>(p)->~INonTrivial();
				// go to next object
				p = reinterpret_cast<INonTrivial*>(reinterpret_cast<char*>(p)+stride);
			}
		}
};

inline IObjectPoolBase::~IObjectPoolBase()
{
}

inline IObjectPoolBase::INonTrivial::~INonTrivial()
{
}


// The simplest of all Garbage Collectors, difference between `CMemoryPool` is that it will run the destructor on all objects it ever allocated when being reset
template<MemoryPoolConfig Config>
class CObjectPool final : public IObjectPoolBase
{
		using this_t = CObjectPool<Config>;
		using addr_allocator_type = Config::AddressAllocator;

    public:
		// Multi-thread safe address allocators shouldn't be used in the MemoryPoolConfigs, because the mem pool needs to sync more stuff
		constexpr static inline bool CanRecycleAllocations = std::is_same_v<LinearAddressAllocatorST<typename addr_allocator_type::size_type>,addr_allocator_type>;
		//
        using mem_pool_type = CMemoryPool<Config>;
		using block_allocator_type = typename mem_pool_type::block_allocator_type;
        using size_type = typename mem_pool_type::size_type;
		//
		template<typename T> requires (std::is_base_of_v<INonTrivial,std::remove_const_t<T>>||std::is_trivially_destructible_v<std::remove_const_t<T>>)
		using typed_pointer_type = mem_pool_type::template typed_pointer_type<T>;

        using creation_params_type = typename mem_pool_type::creation_params_type;
		inline CObjectPool(creation_params_type&& params) : m_pool(std::move(params)) {}
		// Destructor performs a form of garbage collection (just to make sure destructors are ran)
		// NOTE: C++26 reflection would allow us to find all the `Handle` and `TypedHandle<U>` in `T` and do actual mark-and-sweep Garbage Collection
		inline ~CObjectPool()
		{
			for (auto& entry : m_allocations)
				destroy(deref<INonTrivial>(entry.first),entry.second.count,entry.second.stride);
		}

		//
		struct check_t
		{
			bool value;
		};
		template<typename T> requires (!std::is_const_v<T>)
		inline T* deref(const typed_pointer_type<T> h, check_t check={.value=false})
		{
			T* retval = m_pool.deref<T>(h);
			if (!retval)
				return nullptr;
			// check double free
			if constexpr (std::is_base_of_v<INonTrivial,T>)
			if (check.value)
			{
				if (auto found=m_allocations.find(h); found==m_allocations.end())
					return nullptr;
			}
			return retval;
		}
		template<typename T> requires std::is_const_v<T>
		inline T* deref(const typed_pointer_type<T> h, check_t check={.value=false}) const
		{
			using non_const_t = std::remove_const_t<T>;
			const auto mutableH = block_allocator_type::template _const_cast<non_const_t>(h);
			return const_cast<this_t*>(this)->deref<non_const_t>(mutableH,check);
		}

		//
        template <typename T, typename... FuncArgs> requires (!std::is_array_v<T>) // for now until we have a test
        inline typed_pointer_type<T> emplace_n(const size_type n, FuncArgs&&... args)
        {
			if (n>>MaxSingleAllocCountLog2)
				return {};
			if constexpr (Config::ThreadSafe)
				m_pool.getLock().lock();
            //
            constexpr size_type a = alignof(T);
			const size_type size = [&]()->size_type
			{
				// Objects may be variable size, but each gets the same ctor arguments, so they have the same size!
				if constexpr (std::is_base_of_v<IVariableSize,T>)
				{
					const auto size = T::calc_size(args...);
					assert((size>>MaxNonTrivialObjectSizeLog2)==0);
					assert((size&(a-1))==0);
					return size;
				}
				else
					return sizeof(T);
			}();
			//
            const typed_pointer_type<T> retval = m_pool.allocate(size*n,a);
			// record existence if needed
			if constexpr (std::is_base_of_v<INonTrivial,T>)
			{
				static_assert((sizeof(T)>>MaxNonTrivialObjectSizeLog2)==0);
				if (retval)
					m_allocations[retval] = {.stride=size,.count=n};
			}
			else
				static_assert(std::is_trivially_constructible_v<T>,"All non-trivially-constructible `T` must inherit from INonTrivial!");
			if constexpr (Config::ThreadSafe)
				m_pool.getLock().unlock();
			// run constructors
			if constexpr (std::is_base_of_v<INonTrivial,T>)
            if (retval)
			{
				T* ptr = deref<T>(retval);
				for (size_type i=0; i<n; i++)
				{
					if constexpr (sizeof...(FuncArgs)!=0u)
						std::construct_at(ptr,std::forward<FuncArgs>(args)...);
					else
						std::uninitialized_default_construct(ptr);
					// go to next object
					ptr = reinterpret_cast<T*>(reinterpret_cast<char*>(ptr)+size);
				}
            }
            return retval;
        }
        template <typename T, typename... FuncArgs>
        inline typed_pointer_type<T> emplace(FuncArgs&&... args)
        {
            return emplace_n<T,FuncArgs...>(1u,std::forward<FuncArgs>(args)...);
        }
		
		//
        template <typename T> requires (!std::is_array_v<T>) // for now until we have a test
        inline void _delete(const typed_pointer_type<T> h, const size_type n=1)
		{
			if constexpr (Config::ThreadSafe)
				m_pool.getLock().lock();
			// destroy and get total size
            size_type size = static_cast<size_type>(sizeof(T));
			if constexpr (std::is_base_of_v<INonTrivial,T>)
			{
				// remove from our list of live allocations
				auto found = m_allocations.find(h);
				assert(found!=m_allocations.end());
				assert(found->second.count==n);
				if constexpr (std::is_base_of_v<IVariableSize,T>)
					size = found->second.stride;
				destroy(deref<INonTrivial>(h),n,size);
				m_allocations.erase(found);
			}
			else
				static_assert(std::is_trivially_destructible_v<T>,"All non-trivially-destructible `T` must inherit from INonTrivial!");
			// free the memory
			m_pool.deallocate(h,size*n);
			if constexpr (Config::ThreadSafe)
				m_pool.getLock().unlock();
		}

    private:
        CMemoryPool<Config> m_pool;
		// Handle and object count
		constexpr static inline size_type MaxSingleAllocCountLog2 = 5;
		constexpr static inline size_type MaxNonTrivialObjectSizeLog2 = sizeof(size_type)*8-MaxSingleAllocCountLog2;
		struct SMetadata
		{
			size_type stride : MaxNonTrivialObjectSizeLog2;
			size_type count : MaxSingleAllocCountLog2;
		};
		using allocation_record_t = core::unordered_map<typed_pointer_type<INonTrivial>,SMetadata>;
		allocation_record_t m_allocations;
};

}
#endif
