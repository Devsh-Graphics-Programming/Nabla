// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

namespace nbl
{
namespace hlsl
{

namespace tgmath
{

template <typename T>
inline bool isnan(T val)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	using AsFloat = typename float_of_size<sizeof(T)>::type;

	AsUint asUint = bit_cast<AsUint, T>(val);
	return bool((ieee754::extractBiasedExponent<T>(val) == ieee754::traits<AsFloat>::specialValueExp) && (asUint & ieee754::traits<AsFloat>::mantissaMask));
}

// TODO: better implementation, also i'm not sure this is the right place for this function
template<typename T>
NBL_CONSTEXPR_INLINE_FUNC enable_if<is_fundamental<T>::type, T>::type lerp(T a, T b, bool c)
{
	return c ? b : a;
}

template<typename Uint>
NBL_CONSTEXPR_INLINE_FUNC bool isInf(Uint val)
{
	using AsFloat = typename float_of_size<sizeof(Uint)>::type;
	return (val & ~ieee754::traits<AsFloat>::signMask) == ieee754::traits<AsFloat>::inf;
}

}

}
}

#endif