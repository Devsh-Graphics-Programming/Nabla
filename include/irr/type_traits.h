// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_TYPE_TRAITS_H_INCLUDED__
#define __NBL_TYPE_TRAITS_H_INCLUDED__

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
}

#endif

