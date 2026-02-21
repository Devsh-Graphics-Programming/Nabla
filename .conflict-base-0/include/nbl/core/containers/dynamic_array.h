// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_DYNAMIC_ARRAY_H_INCLUDED_
#define _NBL_CORE_DYNAMIC_ARRAY_H_INCLUDED_

#include "nbl/macros.h"
#include "nbl/core/decl/Types.h" //for core::allocator

namespace nbl::core
{

namespace impl
{
	template<class allocator>
	class NBL_FORCE_EBO dynamic_array_typeless_base // : public Uncopyable, public Unmovable // cannot due to diamond inheritance
	{
		protected:
			allocator alctr;
			size_t item_count;

			dynamic_array_typeless_base(const allocator& _alctr, size_t _item_count) : alctr(_alctr), item_count(_item_count) {}

			virtual ~dynamic_array_typeless_base() {} // do not remove, need for class size computation!

		public:
			_NBL_NO_COPY_FINAL(dynamic_array_typeless_base);
			_NBL_NO_MOVE_FINAL(dynamic_array_typeless_base);
	};

	template<class allocator, typename... DataTypeAndOverAlignmentTypes>
	class NBL_FORCE_EBO alignas(ResolveAlignment<dynamic_array_typeless_base<allocator>,AlignedBase<alignof(DataTypeAndOverAlignmentTypes)>...>) dynamic_array_base : public dynamic_array_typeless_base<allocator>
	{
			using Base = dynamic_array_typeless_base<allocator>;

		protected:
			using Base::Base;
	};
}

//! Class for array type, that allocates memory one time dynamically for specified constant amount of objects
/**
	An array is allocated dynamically, not on stack, so compared to std::array its size can be determined at runtime,
	but there is no case in you can change the size of such an array.

	The adventage of this class is that it has constant storage size,
	so only one allocation is performed once compared to std::vector (member and data storage on single allocation),
	instead of unnecessary 2 allocations std::vector performs.

	As a consequence
	
	\code{.cpp}
	sizeof(dynamic_array<T,allocator>) 
	\endcode
	
	is completely meaningless since the size isn't known on compile-time, and it can only be allocated on the heap and is furthermore non-copyable.

	@see core::refctd_dynamic_array
*/
template<typename T, class allocator, class CRTP, typename... OverAlignmentTypes>
class NBL_FORCE_EBO dynamic_array : public impl::dynamic_array_base<allocator,T,OverAlignmentTypes...>
{
		using base_t = impl::dynamic_array_base<allocator,T,OverAlignmentTypes...>;
		_NBL_STATIC_INLINE_CONSTEXPR bool is_const = std::is_const<T>::value;

	public:
		using allocator_type = allocator;
		using value_type = T;
		using pointer = typename std::allocator_traits<allocator_type>::pointer;
		using const_pointer = typename std::allocator_traits<allocator_type>::const_pointer;
		using iterator = T*;
		using const_iterator = const T*;

		using this_real_type = typename std::conditional<std::is_void<CRTP>::value, dynamic_array<T,allocator,void,OverAlignmentTypes...>, CRTP>::type;

	protected:
		// little explainer what is happening here, the derived CRTP class could have some other ancestors before us, so we need static cast to get pointer to CRTP head
		inline auto storage() {return reinterpret_cast<std::remove_const_t<T>*>(static_cast<this_real_type*>(this))+this_real_type::dummy_item_count;}

		inline dynamic_array(size_t _length, const allocator& _alctr = allocator()) : base_t( _alctr,_length )
		{
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::construct(base_t::alctr,storage()+i);
		}
		inline dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator()) : base_t( _alctr,_length )
		{
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::construct(base_t::alctr,storage()+i,_val);
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		inline dynamic_array(const container_t& _containter, const allocator& _alctr = allocator()) : base_t( _alctr,_containter.size())
		{
			auto it = _containter.begin();
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::construct(base_t::alctr,storage()+i,*(it++));
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		inline dynamic_array(container_t&& _containter, const allocator& _alctr = allocator()) : base_t( _alctr,_containter.size())
		{
			auto it = _containter.begin();
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::construct(base_t::alctr,storage()+i,std::move(*(it++)));
		}

	public:
		_NBL_STATIC_INLINE_CONSTEXPR size_t dummy_item_count = (sizeof(base_t)+sizeof(T)-1ull)/sizeof(T);

		virtual ~dynamic_array()
		{
			for (size_t i = 0ull; i < base_t::item_count; ++i)
				std::allocator_traits<allocator>::destroy(base_t::alctr,storage()+i);
		}

		static inline size_t size_of(size_t length)
		{
			return (this_real_type::dummy_item_count + length) * sizeof(T);
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		static inline size_t size_of(const container_t& _containter)
		{
			return (this_real_type::dummy_item_count + _containter.size()) * sizeof(T);
		}

		static inline void* allocate_dynamic_array(size_t length)
		{
            auto gccHappyVar = allocator();
			return std::allocator_traits<allocator>::allocate(gccHappyVar, this_real_type::size_of(length) / sizeof(T));
		}
		static inline void* allocate_dynamic_array(size_t length, const T&)
		{
			return allocate_dynamic_array(length);
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		static inline void* allocate_dynamic_array(const container_t& _containter)
		{
    		auto gccHappyVar = allocator();
				return std::allocator_traits<allocator>::allocate(gccHappyVar, this_real_type::size_of(_containter) / sizeof(T));
		}
		static inline void* allocate_dynamic_array(size_t length, allocator& _alctr)
		{
			return std::allocator_traits<allocator>::allocate(_alctr,this_real_type::size_of(length)/sizeof(T));
		}
		static inline void* allocate_dynamic_array(size_t length, const T&, allocator& _alctr)
		{
			return allocate_dynamic_array(length, _alctr);
		}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		static inline void* allocate_dynamic_array(const container_t& _containter, allocator& _alctr)
		{
			return std::allocator_traits<allocator>::allocate(_alctr, this_real_type::size_of(_containter)/sizeof(T));
		}
		// factory method to use instead of `new`
		template<typename... Args>
		static inline this_real_type* create_dynamic_array(Args&&... args)
		{
			void* ptr = this_real_type::allocate_dynamic_array(args...);
			return new(ptr) this_real_type(std::forward<Args>(args)...);
		}

		// the usual new allocation operator won't let us analyze the constructor arguments to decide the size of the object
		static void* operator new(size_t size) = delete;
		// we cannot new or delete arrays of `dynamic_array` (non uniform sizes)
		static void* operator new[](size_t size) = delete;

        static inline void operator delete(void* ptr) noexcept
        {
			allocator().deallocate(reinterpret_cast<pointer>(ptr));
        }


		// size hint is ill-formed
		static void operator delete(void* ptr, std::size_t sz) = delete;
		// no arrays
		static void operator delete[](void* ptr) = delete;
		static void operator delete[](void* ptr, std::size_t sz) = delete;
		static void* operator new(size_t size, std::align_val_t al) = delete;
		static void* operator new[](size_t size, std::align_val_t al) = delete;

		// TODO: Support for C++17 align allocator
		static void operator delete(void* ptr, std::align_val_t al) = delete;
		static void operator delete(void* ptr, std::size_t sz, std::align_val_t al) = delete;
		// no arrays
		static void operator delete[](void* ptr, std::align_val_t al) = delete;
		static void operator delete[](void* ptr, std::size_t sz, std::align_val_t al) = delete;

		inline bool operator!=(const this_real_type& _other) const
		{
			if (size() != _other.size())
				return true;
			for (size_t i = 0u; i < size(); ++i)
				if ((*this)[i] != _other[i])
					return true;
			return false;
		}
		inline bool operator==(const this_real_type& _other) const
		{
			return !((*this) != _other);
		}

		template<typename U=T> requires (!std::is_same_v<iterator,const_iterator>)
		inline iterator			begin() noexcept { return data(); }
		inline const_iterator	begin() const noexcept { return data(); }
		template<typename U=T> requires (!std::is_same_v<iterator,const_iterator>)
		inline iterator			end() noexcept { return data()+size(); }
		inline const_iterator	end() const noexcept { return data()+size(); }
		inline const_iterator	cend() const noexcept { return data()+size(); }
		inline const_iterator	cbegin() const noexcept { return data(); }

		inline size_t			size() const noexcept { return base_t::item_count; }
		inline bool				empty() const noexcept { return !size(); }

		inline size_t			bytesize() const noexcept { return base_t::item_count*sizeof(T); }

		inline const T&			operator[](size_t ix) const noexcept { return data()[ix]; }
		template<typename U=T> requires (!std::is_same_v<iterator,const_iterator>)
		inline T&				operator[](size_t ix) noexcept { return data()[ix]; }
		
		template<typename U=T> requires (!std::is_same_v<iterator,const_iterator>)
		inline T&				front() noexcept { return *begin(); }
		inline const T&			front() const noexcept { return *begin(); }
		template<typename U=T> requires (!std::is_same_v<iterator,const_iterator>)
		inline T&				back() noexcept { return *(end()-1); }
		inline const T&			back() const noexcept { return *(end()-1); }
		template<typename U=T> requires (!std::is_same_v<iterator,const_iterator>)
		inline pointer			data() noexcept { return storage(); }
		inline const_pointer	data() const noexcept { return reinterpret_cast<const_pointer>(static_cast<const this_real_type*>(this))+this_real_type::dummy_item_count; }
};


}

#endif