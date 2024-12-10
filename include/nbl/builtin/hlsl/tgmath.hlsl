// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#endif

namespace nbl
{
namespace hlsl
{
namespace tgmath
{

template <typename T>
inline bool isNaN(T val)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	using AsFloat = typename float_of_size<sizeof(T)>::type;

	AsUint asUint = bit_cast<AsUint, T>(val);
	return bool((ieee754::extractBiasedExponent<T>(val) == ieee754::traits<AsFloat>::specialValueExp) && (asUint & ieee754::traits<AsFloat>::mantissaMask));
}

template<typename T>
inline bool isInf(T val)
{
	using AsUint = typename unsigned_integer_of_size<sizeof(T)>::type;
	using AsFloat = typename float_of_size<sizeof(T)>::type;

	AsUint tmp = bit_cast<AsUint>(val);
	return (tmp & (~ieee754::traits<AsFloat>::signMask)) == ieee754::traits<AsFloat>::inf;
}

#ifdef __HLSL_VERSION
#define DEFINE_IS_NAN_SPECIALIZATION(TYPE)\
template<>\
inline bool isNaN<TYPE>(TYPE val)\
{\
	return spirv::isnan(val);\
}\

#define DEFINE_IS_INF_SPECIALIZATION(TYPE)\
template<>\
inline bool isInf<TYPE>(TYPE val)\
{\
	return spirv::isinf(val);\
}\

DEFINE_IS_NAN_SPECIALIZATION(float16_t)
DEFINE_IS_NAN_SPECIALIZATION(float32_t)
DEFINE_IS_NAN_SPECIALIZATION(float64_t)

DEFINE_IS_INF_SPECIALIZATION(float16_t)
DEFINE_IS_INF_SPECIALIZATION(float32_t)
DEFINE_IS_INF_SPECIALIZATION(float64_t)

#undef DEFINE_IS_INF_SPECIALIZATION
#undef DEFINE_IS_NAN_SPECIALIZATION
#undef INTRINSIC_FUNC_NAMESPACE
#endif

}

}
}

#endif