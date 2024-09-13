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
namespace impl
{

template<typename T, bool IsTFundamental = is_fundamental<T>::value>
NBL_CONSTEXPR_INLINE_FUNC bool isInf(T val)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	using AsFloat = typename float_of_size<sizeof(T)>::type;

	if (IsTFundamental)
	{
		return isinf(bit_cast<AsFloat>(val));
	}
	else
	{
		AsUint tmp = bit_cast<AsUint>(val);
		return (tmp & (~ieee754::traits<AsFloat>::signMask)) == ieee754::traits<AsFloat>::inf;
	}
}

}

template <typename T>
inline bool isNaN(T val)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	using AsFloat = typename float_of_size<sizeof(T)>::type;

	AsUint asUint = bit_cast<AsUint, T>(val);
	return bool((ieee754::extractBiasedExponent<T>(val) == ieee754::traits<AsFloat>::specialValueExp) && (asUint & ieee754::traits<AsFloat>::mantissaMask));
}

template<typename T>
NBL_CONSTEXPR_INLINE_FUNC bool isInf(T val)
{
	return impl::isInf(val);
}

}

}
}

#endif