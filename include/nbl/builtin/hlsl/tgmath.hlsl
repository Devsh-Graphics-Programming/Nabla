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
// TODO: will not work for emulated_float as an input because `concepts::floating_point<T>` is only for native floats, fix every occurance
template<typename FloatingPoint NBL_FUNC_REQUIRES(concepts::FloatingPointLike<FloatingPoint>)
inline FloatingPoint erf(FloatingPoint x)
{
    return tgmath_impl::erf_helper<FloatingPoint>::__call(x);
}

template<typename FloatingPoint NBL_FUNC_REQUIRES(concepts::FloatingPointLike<FloatingPoint>)
inline FloatingPoint erfInv(FloatingPoint x)
{
    return tgmath_impl::erfInv_helper<FloatingPoint>::__call(x);
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLike<T> || concepts::Vectorial<T>)
inline T floor(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::floor_helper<T>::__call(val);
}

template<typename T, typename U NBL_FUNC_REQUIRES((concepts::FloatingPointLike<T> || concepts::FloatingPointLikeVectorial<T>) && (concepts::floating_point<U> || is_same_v<U, bool>))
inline T lerp(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(U) a)
{
    return tgmath_impl::lerp_helper<T, U>::__call(x, y, a);
}

template<typename FloatingPoint NBL_FUNC_REQUIRES(concepts::floating_point<FloatingPoint>)
inline bool isnan(NBL_CONST_REF_ARG(FloatingPoint) val)
{
    return tgmath_impl::isnan_helper<FloatingPoint>::__call(val);
}

template<typename V NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<V>)
inline vector<bool, vector_traits<V>::Dimension> isnan(NBL_CONST_REF_ARG(V) val)
{
    return tgmath_impl::isnan_helper<V>::__call(val);
}

template<typename FloatingPoint NBL_FUNC_REQUIRES(concepts::floating_point<FloatingPoint>)
inline FloatingPoint isinf(NBL_CONST_REF_ARG(FloatingPoint) val)
{
    return tgmath_impl::isinf_helper<FloatingPoint>::__call(val);
}

template<typename V NBL_FUNC_REQUIRES(concepts::FloatingPointLikeVectorial<V>)
inline vector<bool, vector_traits<V>::Dimension> isinf(NBL_CONST_REF_ARG(V) val)
{
    return tgmath_impl::isinf_helper<V>::__call(val);
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLike<T> || concepts::FloatingPointLikeVectorial<T>)
inline T pow(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y)
{
    return tgmath_impl::pow_helper<T>::__call(x, y);
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLike<T> || concepts::FloatingPointLikeVectorial<T>)
inline T exp(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::exp_helper<T>::__call(x);
}


template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLike<T> || concepts::Vectorial<T>)
inline T exp2(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::exp2_helper<T>::__call(x);
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLike<T> || concepts::FloatingPointLikeVectorial<T>)
inline T log(NBL_CONST_REF_ARG(T) x)
{
    return tgmath_impl::log_helper<T>::__call(x);
}

template<typename T NBL_FUNC_REQUIRES(concepts::FloatingPointLike<T> || concepts::signed_integral<T> || concepts::FloatingPointLikeVectorial<T> || concepts::SignedIntVectorial<T>)
inline T abs(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::abs_helper<T>::__call(val);
}

template<typename T NBL_FUNC_REQUIRES(concepts::floating_point<T> || concepts::Vectorial<T>)
inline T sqrt(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::sqrt_helper<T>::__call(val);
}

template<typename T NBL_FUNC_REQUIRES(concepts::floating_point<T> || concepts::Vectorial<T>)
inline T sin(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::sin_helper<T>::__call(val);
}

template<typename T NBL_FUNC_REQUIRES(concepts::floating_point<T> || concepts::Vectorial<T>)
inline T cos(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::cos_helper<T>::__call(val);
}

template<typename T NBL_FUNC_REQUIRES(concepts::floating_point<T> || concepts::Vectorial<T>)
inline T acos(NBL_CONST_REF_ARG(T) val)
{
    return tgmath_impl::acos_helper<T>::__call(val);
}

}
}

#endif
