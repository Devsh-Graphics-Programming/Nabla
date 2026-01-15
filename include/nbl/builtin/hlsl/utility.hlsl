// Copyright (C) 2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_UTILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILITY_INCLUDED_


#include <nbl/builtin/hlsl/type_traits.hlsl>


namespace nbl
{
namespace hlsl
{

template<typename T1, typename T2>
struct pair
{
	using first_type = T1;
	using second_type = T2;

	first_type first;
	second_type second;
};

template<typename T1, typename T2>
pair<T1, T2> make_pair(T1 f, T2 s)
{
	pair<T1, T2> p;
	p.first = f;
	p.second = s;
	return p;
}

template<typename T1, typename T2>
void swap(NBL_REF_ARG(pair<T1, T2>) a, NBL_REF_ARG(pair<T1, T2>) b)
{
	T1 temp_first = a.first;
	T2 temp_second = a.second;
	a.first = b.first;
	a.second = b.second;
	b.first = temp_first;
	b.second = temp_second;
}

template<typename T>
const static bool always_true = true;
#ifndef __HLSL_VERSION

template<class T>
std::add_rvalue_reference_t<T> declval() noexcept
{
	static_assert(false,"Actually calling declval is ill-formed.");
}

#else

namespace experimental
{

template<class T>
T declval() {}

}

#endif
}
}

#endif
