// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_

#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

namespace nbl
{
namespace hlsl
{
template <typename Float>
bool isnan(Float val)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(Float)>::type;
	using AsFloat = typename float_of_size<sizeof(Float)>::type;
	AsUint asUint = bit_cast<AsUint, Float>(val);
	return bool((ieee754::extractBiasedExponent<Float>(val) == ieee754::traits<Float>::specialValueExp) && (asUint & ieee754::traits<Float>::mantissaMask));
}

}
}

#endif