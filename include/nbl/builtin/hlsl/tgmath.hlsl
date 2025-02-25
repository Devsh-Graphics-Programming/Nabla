// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_

#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/tgmath/impl.hlsl>
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

template<typename T>
inline T tan(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::tan_helper<T>::__call(val);
}

template<typename T>
inline T asin(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::asin_helper<T>::__call(val);
}

template<typename T>
inline T atan(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::atan_helper<T>::__call(val);
}

template<typename T>
inline T sinh(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::sinh_helper<T>::__call(val);
}

template<typename T>
inline T cosh(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::cosh_helper<T>::__call(val);
}

template<typename T>
inline T tanh(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::tanh_helper<T>::__call(val);
}

template<typename T>
inline T asinh(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::asinh_helper<T>::__call(val);
}

template<typename T>
inline T acosh(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::acosh_helper<T>::__call(val);
}

template<typename T>
inline T atanh(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::atanh_helper<T>::__call(val);
}

template<typename T>
inline T atan2(NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::atan2_helper<T>::__call(y, x);
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

template<typename T>
inline T round(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::round_helper<T>::__call(val);
}

template<typename T>
inline T roundEven(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::roundEven_helper<T>::__call(val);
}

template<typename T>
inline T trunc(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::trunc_helper<T>::__call(val);
}

template<typename T>
inline T ceil(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::ceil_helper<T>::__call(val);
}

template<typename T>
inline T fma(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(T) z)
{
    return tgmath_impl::fma_helper<T>::__call(x, y, z);
}

template<typename T, typename U>
inline T ldexp(NBL_CONST_REF_ARG(T) arg, NBL_CONST_REF_ARG(U) exp)
{
    return tgmath_impl::ldexp_helper<T, U>::__call(arg, exp);
}

template<typename T>
inline ModfOutput<T> modfStruct(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::modfStruct_helper<T>::__call(val);
}

template<typename T>
inline FrexpOutput<T> frexpStruct(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::frexpStruct_helper<T>::__call(val);
}

}
}

#endif
