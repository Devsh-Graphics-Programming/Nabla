// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_REFCTD_DYNAMIC_ARRAY_H_INCLUDED_
#define _NBL_CORE_REFCTD_DYNAMIC_ARRAY_H_INCLUDED_

#include "nbl/core/decl/Types.h"
#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/alloc/AlignedBase.h"
#include "nbl/core/containers/dynamic_array.h"

namespace nbl::core
{

//! Class for array type, that allocates memory one time dynamically for specified constant amount of objects
/**
	An array is allocated dynamically, not on stack, so compared to std::array its size can be determined at runtime,
	but there is no case in you can change the size of such an array.

	The adventage of this class is that it has constant storage size,
	so only one allocation is performed once compared to std::vector (member and data storage on single allocation),
	instead of unnecessary 2 allocations std::vector performs.

	As a consequence
	
	\code{.cpp}
	sizeof(refctd_dynamic_array<T,allocator>) 
	\endcode
	
	is completely meaningless since the size isn't known on compile-time, and it can only be allocated on the heap and is furthermore non-copyable.

	The purpose of this class is to compensate for the non-copyability of the base class compared to core::dynamic_array
	and allow "pass by reference" (shared contents) without memory leaks and going out of scope.

	@see IReferenceCounted
	@see core::dynamic_array
*/
template<typename T, class allocator=core::allocator<typename std::remove_const<T>::type>, typename... OverAlignmentTypes>
class NBL_FORCE_EBO refctd_dynamic_array : public IReferenceCounted, public dynamic_array<T,allocator,refctd_dynamic_array<T,allocator,OverAlignmentTypes...>,OverAlignmentTypes...>
{
	public:
		using this_type = refctd_dynamic_array<T,allocator,OverAlignmentTypes...>;
		using const_type = refctd_dynamic_array<const T,allocator,OverAlignmentTypes...>;

		friend class dynamic_array<T,allocator,this_type,OverAlignmentTypes...>;
		using base_t = dynamic_array<T,allocator,this_type,OverAlignmentTypes...>;
		//friend class base_t; // won't work

		using meta_base_t = dynamic_array<T,allocator,void,OverAlignmentTypes...>;
		static_assert(sizeof(base_t) == sizeof(meta_base_t), "non-CRTP and CRTP base class definitions differ in size");
		static_assert(sizeof(meta_base_t) == sizeof(impl::dynamic_array_base<allocator,T,OverAlignmentTypes...>), "memory has been added to dynamic_array"); // TODO: fix

		class NBL_FORCE_EBO fake_size_class : public IReferenceCounted, meta_base_t {
			using meta_base_t::operator delete;
		};

	public:
		_NBL_STATIC_INLINE_CONSTEXPR size_t dummy_item_count = (sizeof(fake_size_class)+sizeof(T)-1ull)/sizeof(T);

		_NBL_RESOLVE_NEW_DELETE_AMBIGUITY(base_t) // only want new and delete operators from `dynamic_array`

		virtual ~refctd_dynamic_array() = default; // would like to move to `protected`

		inline operator const_type&()
		{
			return *reinterpret_cast<const_type*>(this);
		}
		inline operator const const_type&() const
		{
			return *reinterpret_cast<const const_type*>(this);
		}
		inline operator const_type&&() &&
		{
			return std::move(*reinterpret_cast<const_type*>(this));
		}

	protected:
		inline refctd_dynamic_array(size_t _length, const allocator& _alctr = allocator()) : base_t(_length, _alctr) {}
		inline refctd_dynamic_array(size_t _length, const T& _val, const allocator& _alctr = allocator()) : base_t(_length, _val, _alctr) {}
		inline refctd_dynamic_array(const this_type& _containter) : base_t(_containter, _containter.alctr) {}
		inline refctd_dynamic_array(const this_type& _containter, const allocator& _alctr) : base_t(_containter, _alctr) {}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		inline refctd_dynamic_array(const container_t& _containter, const allocator& _alctr = allocator()) : base_t(_containter, _alctr) {}
		template<typename container_t, typename iterator_t = typename container_t::iterator>
		inline refctd_dynamic_array(container_t&& _containter, const allocator& _alctr = allocator()) : base_t(std::move(_containter),_alctr) {}
};


template<typename T, class allocator=core::allocator<typename std::remove_const<T>::type>, typename... OverAlignmentTypes>
using smart_refctd_dynamic_array = smart_refctd_ptr<refctd_dynamic_array<T,allocator,OverAlignmentTypes...> >;

template<class smart_refctd_dynamic_array_type, typename... Args>
inline smart_refctd_dynamic_array_type make_refctd_dynamic_array(Args&&... args)
{
	using srdat = typename std::remove_const<smart_refctd_dynamic_array_type>::type;
	return srdat(srdat::pointee::create_dynamic_array(std::forward<Args>(args)...),dont_grab);
}


}

#endif