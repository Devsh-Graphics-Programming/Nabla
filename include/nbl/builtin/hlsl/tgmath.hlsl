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
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/concepts/vector.hlsl>
// C++ headers
#ifndef __HLSL_VERSION
#include <algorithm>
#include <cmath>
#endif

namespace nbl
{
namespace hlsl
{
template<typename T>
inline T erf(T x)
{
    return tgmath_impl::erf_helper<T>::__call(x);
}

template<typename T>
inline T erfInv(T x)
{
    return tgmath_impl::erfInv_helper<T>::__call(x);
}

template<typename T>
inline T floor(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::floor_helper<T>::__call(val);
}

template<typename T, typename U>
inline T mix(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(U) a)
{
    return tgmath_impl::mix_helper<T, U>::__call(x, y, a);
}

template<typename T>
inline typename tgmath_impl::isnan_helper<T>::return_t isnan(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::isnan_helper<T>::__call(val);
}

template<typename T>
inline typename tgmath_impl::isinf_helper<T>::return_t isinf(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::isinf_helper<T>::__call(val);
}

template<typename T>
inline T pow(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y)
{
    return tgmath_impl::pow_helper<T>::__call(x, y);
}

template<typename T>
inline T exp(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::exp_helper<T>::__call(x);
}

template<typename T>
inline T exp2(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::exp2_helper<T>::__call(x);
}

template<typename T>
inline T log(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::log_helper<T>::__call(x);
}

template<typename T>
inline T log2(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::log2_helper<T>::__call(x);
}

template<typename T>
inline T abs(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::abs_helper<T>::__call(val);
}

template<typename T>
inline T sqrt(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::sqrt_helper<T>::__call(val);
}

template<typename T>
inline T sin(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::sin_helper<T>::__call(val);
}

template<typename T>
inline T cos(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::cos_helper<T>::__call(val);
}

template<typename T>
inline T acos(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::acos_helper<T>::__call(val);
}

/**
* @brief Returns fractional part of given floating-point value.
*
* @tparam T type of the value to operate on.
*
* @param [in] val The value to retrive fractional part from.
*/
template<typename T>
inline T modf(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::modf_helper<T>::__call(val);
}

}
}

#endif
