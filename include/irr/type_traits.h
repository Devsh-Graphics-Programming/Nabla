// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h
#ifndef __IRR_TYPE_TRAITS_H_INCLUDED__
#define __IRR_TYPE_TRAITS_H_INCLUDED__

#include <type_traits>

namespace irr
{

#if __cplusplus >= 201703L
template<bool B>
using bool_constant = std::bool_constant<B>;
#else
template<bool B>
using bool_constant = std::integral_constant<bool, B>;
#endif // C


template<typename T, typename U, typename... Us>
struct is_any_of : std::integral_constant<bool,
	std::conditional<
		std::is_same<T, U>::value,
		std::true_type,
		irr::is_any_of<T, Us...>
	>::type::value
>
{};

template<typename T, typename U>
struct is_any_of<T, U> : std::is_same<T, U>::type { };



template<typename T, bool is_const_pointer = std::is_const_v<T>>
struct ptr_to_const;

template<typename T>
struct ptr_to_const<T, false>
{
	static_assert(std::is_pointer_v<T>);

	using type = std::add_pointer_t<
		std::add_const_t<
			std::remove_pointer_t<T>
		>
	>;
};

template<typename T>
struct ptr_to_const<T, true>
{
	using type = std::add_const_t<typename ptr_to_const<T, false>::type>;
};

template<typename T>
using ptr_to_const_t = typename ptr_to_const<T>::type;


template<typename T, bool is_pointer = std::is_pointer_v<T>>
struct pointer_level_count;

template<typename T>
struct pointer_level_count<T, false> : std::integral_constant<int, 0> {};

template<typename T>
struct pointer_level_count<T, true> : std::integral_constant<int, pointer_level_count<std::remove_pointer_t<T>>::value + 1> {};

template<typename T>
inline constexpr int pointer_level_count_v = pointer_level_count<T>::value;


template<typename T, int levels, bool = (levels == 0) || !std::is_pointer_v<T>>
struct remove_pointer_levels;

template<typename T, int levels>
struct remove_pointer_levels<T, levels, true>
{
	using type = T;
};

template<typename T, int levels>
struct remove_pointer_levels<T, levels, false>
{
	static_assert(levels <= pointer_level_count_v<T>);

	using type = typename remove_pointer_levels<std::remove_pointer_t<T>, levels - 1>::type;
};

template<typename T, int levels>
using remove_pointer_levels_t = typename remove_pointer_levels<T, levels>::type;


template<typename T>
using remove_all_pointer_levels = remove_pointer_levels<T, pointer_level_count_v<T>>;

template<typename T>
using remove_all_pointer_levels_t = typename remove_all_pointer_levels<T>::type;


template<typename T>
struct is_pointer_to_const_object : 
	std::bool_constant<
		std::is_const_v<
			remove_all_pointer_levels_t<
				T
			>
		>
	> 
{};

template<typename T>
inline constexpr bool is_pointer_to_const_object_v = is_pointer_to_const_object<T>::value;


template<typename T, int levels>
struct add_pointers
{
	using type = typename add_pointers<T, levels - 1>::type*;
};

template<typename T>
struct add_pointers<T, 0>
{
	using type = T;
};

template<typename T, int levels>
using add_pointers_t = typename add_pointers<T, levels>::type;


// TODO preserve each const-ness of each pointer level
//! type is `const T` in case when T is not a pointer type
//!		or `const T***` in case of `T***`, etc. (preserves pointer depth)
template<typename T>
struct pointer_to_const_object
{
private:
	using object_t = remove_all_pointer_levels_t<T>;
	inline constexpr static int levels = pointer_level_count_v<T>;

public:
	using type = add_pointers_t<const object_t, levels>;
};

template<typename T>
using pointer_to_const_object_t = typename pointer_to_const_object<T>::type;


template<typename T>
struct pointer_to_nonconst_object
{
private:
	using object_t = remove_all_pointer_levels_t<T>;
	inline constexpr static int levels = pointer_level_count_v<T>;

public:
	using type = add_pointers_t<std::remove_const_t<object_t>, levels>;
};

template<typename T>
using pointer_to_nonconst_object_t = typename pointer_to_nonconst_object<T>::type;

}

#endif

