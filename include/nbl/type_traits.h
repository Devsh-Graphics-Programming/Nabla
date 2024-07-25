// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_TYPE_TRAITS_H_INCLUDED__
#define __NBL_TYPE_TRAITS_H_INCLUDED__

#include <type_traits>
#include <array>

// TODO: Unify with the HLSL Header!
namespace nbl
{

template<bool B>
using bool_constant = std::bool_constant<B>;


template <bool... Vals>
using bool_sequence = std::integer_sequence<bool, Vals...>;

template <typename T, typename... Us>
inline constexpr bool is_any_of_v = (... || std::is_same<T, Us>::value);

template<typename T, typename... Us>
struct is_any_of : std::integral_constant<bool, is_any_of_v<T, Us...>> {};

static_assert(is_any_of<bool, bool>::value == true, "is_any_of test");
static_assert(is_any_of<bool, int>::value == false, "is_any_of test");
static_assert(is_any_of<bool, int, bool>::value == true, "is_any_of test");
static_assert(is_any_of<bool, int, double>::value == false, "is_any_of test");
static_assert(is_any_of<bool, int, double, bool>::value == true, "is_any_of test");
static_assert(is_any_of<bool, int, double, float>::value == false, "is_any_of test");

template<auto cf, decltype(cf) cmp, decltype(cf)... searchtab>
struct is_any_of_values : is_any_of_values<cf,searchtab...> {};

template<auto cf, decltype(cf) cmp>
struct is_any_of_values<cf, cmp> : std::false_type {}; //if last comparison is also false, than return false

template<auto cf, decltype(cf)... searchtab>
struct is_any_of_values<cf, cf, searchtab...> : std::true_type {};


template<typename T, bool is_const_pointer = std::is_const_v<T>>
struct pointer_to_const;

template<typename T>
struct pointer_to_const<T, false>
{
	static_assert(std::is_pointer_v<T>);

	using type = std::add_pointer_t<
			std::add_const_t<
				std::remove_pointer_t<T>
			>
		>;
};

template<typename T>
struct pointer_to_const<T, true>
{
	using type = std::add_const_t<typename pointer_to_const<T, false>::type>;
};

template<typename T>
using pointer_to_const_t = typename pointer_to_const<T>::type;


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


namespace impl
{

	template <typename... Ts>
	struct type_sequence {};

	template<typename T, bool is_ptr = std::is_pointer_v<T>, typename... Ts>
	struct pointer_level_constness_help;
	template<typename T, typename... Ts>
	struct pointer_level_constness_help<T, false, Ts...>
	{
		using levels_t = type_sequence<Ts...>;
	};
	template<typename T, typename... Ts>
	struct pointer_level_constness_help<T, true, Ts...> : pointer_level_constness_help<std::remove_pointer_t<T>, std::is_pointer_v<std::remove_pointer_t<T>>, Ts..., T> {};


	template<typename T>
	class pointer_level_constness_seq
	{
		template<typename... Levels>
		static constexpr auto get_constness(type_sequence<Levels...>) -> bool_sequence<std::is_const_v<Levels>...>;

	public:
		using constness_seq = decltype(get_constness(typename pointer_level_constness_help<T>::levels_t{}));
	};

	template<typename T, T... Seq>
	struct integer_array
	{
		using type = std::array<T, sizeof...(Seq)>;

		constexpr static type get_array() { return type{ {Seq...} }; }
	};

} // namespace impl

template<typename T>
class pointer_levels_constness
{
	template<bool... Vals>
	constexpr static auto func(bool_sequence<Vals...>) { return impl::integer_array<bool, Vals...>::get_array(); }

public:
	inline static constexpr auto value = func(typename impl::pointer_level_constness_seq<T>::constness_seq{});
};

template<typename T>
inline constexpr auto pointer_levels_constness_v = pointer_levels_constness<T>::value;

namespace impl
{
	template<typename T, int levels, bool is_const_level, bool... is_const_rest>
	struct add_pointers
	{
	private:
		using tmp_type_ = typename add_pointers<
			T,
			levels - 1,
			is_const_rest...
		>::type;

	public:
		using type = std::conditional_t<is_const_level, tmp_type_* const, tmp_type_*>;
	};

	template<typename T, bool is_const_level>
	struct add_pointers<T, 1, is_const_level>
	{
		using type = std::conditional_t<is_const_level, T* const, T*>;
	};

	template<typename T, int levels, bool... is_const_levels>
	using add_pointers_t = typename add_pointers<T, levels, is_const_levels...>::type;

	template<typename T, int levels, bool... constness>
	constexpr auto add_pointers_with_constness_f(bool_sequence<constness...>) -> add_pointers_t<T, levels, constness...>;
} // namespace impl

template<typename T, int levels, bool... constness>
struct add_pointers
{
	using type = impl::add_pointers_t<T, levels, constness...>;
};

template<typename T, int levels, bool... constness>
using add_pointers_t = typename add_pointers<T, levels, constness...>::type;

namespace impl
{
	template<typename T, int levels, typename U>
	struct add_pointers_restore_constness
	{
		using type = decltype(impl::add_pointers_with_constness_f<T, levels>(typename impl::pointer_level_constness_seq<U>::constness_seq{}));
	};
	template<typename T, int levels, typename U>
	using add_pointers_restore_constness_t = typename add_pointers_restore_constness<T, levels, U>::type;
}

//! type is `const T` in case when T is not a pointer type
//!		or `const T**const*` in case of `T**const*`, etc. (preserves pointer depth and constness of each level)
template<typename T>
struct pointer_to_const_object
{
private:
	using object_t = remove_all_pointer_levels_t<T>;
	inline constexpr static int levels = pointer_level_count_v<T>;

public:
	using type = impl::add_pointers_restore_constness_t<const object_t, levels, T>;
};

template<typename T>
using pointer_to_const_object_t = typename pointer_to_const_object<T>::type;


//! Analogous to pointer_to_const_object
template<typename T>
struct pointer_to_nonconst_object
{
private:
	using object_t = remove_all_pointer_levels_t<T>;
	inline constexpr static int levels = pointer_level_count_v<T>;

public:
	using type = impl::add_pointers_restore_constness_t<std::remove_const_t<object_t>, levels, T>;
};

template<typename T>
using pointer_to_nonconst_object_t = typename pointer_to_nonconst_object<T>::type;

}

#endif

