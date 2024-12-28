// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_

#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/impl/tgmath_impl.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/vector_utils/vector_traits.hlsl>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
// C++ headers
#ifndef __HLSL_VERSION
#include <algorithm>
#include <cmath>
#endif

namespace nbl
{
namespace hlsl
{
template<typename FloatingPoint>
inline FloatingPoint erf(FloatingPoint x)
{
    return tgmath_impl::erf_helper<FloatingPoint>::__call(x);
}

template<typename FloatingPoint>
inline FloatingPoint erfInv(FloatingPoint x)
{
    return tgmath_impl::erfInv_helper<FloatingPoint>::__call(x);
}

template<typename T>
inline T floor(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::floor_helper<T>::__call(val);
}

template<typename T, typename U>
inline T lerp(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(U) a)
{
    return tgmath_impl::lerp_helper<T, U>::__call(x, y, a);
}

template<typename FloatingPoint NBL_FUNC_REQUIRES(hlsl::is_floating_point_v<FloatingPoint>)
inline bool isnan(NBL_CONST_REF_ARG(FloatingPoint) val)
{
#ifdef __HLSL_VERSION
    return spirv::isNan<FloatingPoint>(val);
#else
    // GCC and Clang will always return false with call to std::isnan when fast math is enabled,
    // this implementation will always return appropriate output regardless is fas math is enabled or not
    using AsUint = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;
    return tgmath_impl::isnan_uint_impl(reinterpret_cast<const AsUint&>(val));
#endif
}

template<typename FloatingPoint NBL_FUNC_REQUIRES(hlsl::is_floating_point_v<FloatingPoint>)
inline FloatingPoint isinf(NBL_CONST_REF_ARG(FloatingPoint) val)
{
#ifdef __HLSL_VERSION
    return spirv::isInf<FloatingPoint>(val);
#else
    // GCC and Clang will always return false with call to std::isinf when fast math is enabled,
    // this implementation will always return appropriate output regardless is fas math is enabled or not
    using AsUint = typename unsigned_integer_of_size<sizeof(FloatingPoint)>::type;
    return tgmath_impl::isinf_uint_impl(reinterpret_cast<const AsUint&>(val));
#endif
}

template<typename  T>
inline T pow(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y)
{
    return tgmath_impl::pow_helper<T>::__call(x, y);
}

template<typename  T>
inline T exp(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::exp_helper<T>::__call(x);
}


template<typename T>
inline T exp2(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::exp2_helper<T>::__call(x);
}

template<typename  T>
inline T log(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::log_helper<T>::__call(x);
}

template<typename  T>
inline T abs(NBL_CONST_REF_ARG(T) val)
{
#ifdef __HLSL_VERSION
    return abs(val);
#else
    return glm::abs(val);
#endif
}

template<typename  T>
inline T sqrt(NBL_CONST_REF_ARG(T) val)
{
#ifdef __HLSL_VERSION
    return sqrt(val);
#else
    return std::sqrt(val);
#endif
}

template<typename  T>
inline T sin(NBL_CONST_REF_ARG(T) val)
{
#ifdef __HLSL_VERSION
    return sin(val);
#else
    return std::sin(val);
#endif
}

template<typename  T>
inline T cos(NBL_CONST_REF_ARG(T) val)
{
#ifdef __HLSL_VERSION
    return cos(val);
#else
    return std::cos(val);
#endif
}

template<typename  T>
inline T acos(NBL_CONST_REF_ARG(T) val)
{
#ifdef __HLSL_VERSION
    return acos(val);
#else
    return std::acos(val);
#endif
}

}
}

#endif
