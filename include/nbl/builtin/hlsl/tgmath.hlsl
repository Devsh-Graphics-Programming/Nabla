// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/impl/intrinsics_impl.hlsl>
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
inline FloatingPoint erf(FloatingPoint _x)
{
#ifdef __HLSL_VERSION
    const FloatingPoint a1 = 0.254829592;
    const FloatingPoint a2 = -0.284496736;
    const FloatingPoint a3 = 1.421413741;
    const FloatingPoint a4 = -1.453152027;
    const FloatingPoint a5 = 1.061405429;
    const FloatingPoint p = 0.3275911;

    FloatingPoint sign = sign(_x);
    FloatingPoint x = abs(_x);
    
    FloatingPoint t = 1.0 / (1.0 + p*x);
    FloatingPoint y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    
    return sign * y;
#else
    return std::erf(_x);
#endif
}

template<typename FloatingPoint>
inline FloatingPoint erfInv(FloatingPoint _x)
{
    FloatingPoint x = clamp<FloatingPoint>(_x, -0.99999, 0.99999);
#ifdef __HLSL_VERSION
    FloatingPoint w = -log((1.0-x) * (1.0+x));
#else
    FloatingPoint w = -std::log((1.0-x) * (1.0+x));
#endif
    FloatingPoint p;
    if (w<5.0)
    {
        w -= 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p*w;
        p = -3.5233877e-06 + p*w;
        p = -4.39150654e-06 + p*w;
        p = 0.00021858087 + p*w;
        p = -0.00125372503 + p*w;
        p = -0.00417768164 + p*w;
        p = 0.246640727 + p*w;
        p = 1.50140941 + p*w;
    }
    else
    {
#ifdef __HLSL_VERSION
        w = sqrt(w) - 3.0;
#else
        w = std::sqrt(w) - 3.0;
#endif
        p = -0.000200214257;
        p = 0.000100950558 + p*w;
        p = 0.00134934322 + p*w;
        p = -0.00367342844 + p*w;
        p = 0.00573950773 + p*w;
        p = -0.0076224613 + p*w;
        p = 0.00943887047 + p*w;
        p = 1.00167406 + p*w;
        p = 2.83297682 + p*w;
    }
    return p*x;
}


template<typename T>
inline T floor(NBL_CONST_REF_ARG(T) val)
{
#ifdef __HLSL_VERSION
    return spirv::floor(val);
#else
    return glm::floor(val);
#endif
    
}

template<typename T, typename U>
inline T lerp(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y, NBL_CONST_REF_ARG(U) a)
{
    return cpp_compat_intrinsics_impl::lerp_helper<T, U>::lerp(x, y, a);
}

template<typename FloatingPoint>
inline FloatingPoint isnan(NBL_CONST_REF_ARG(FloatingPoint) val)
{
#ifdef __HLSL_VERSION
    return spirv::isNan(val);
#else
    return std::isnan(val);
#endif
}

template<typename FloatingPoint>
inline FloatingPoint isinf(NBL_CONST_REF_ARG(FloatingPoint) val)
{
#ifdef __HLSL_VERSION
    return spirv::isInf(val);
#else
    return std::isinf(val);
#endif
}

template<typename  T>
inline T exp2(NBL_CONST_REF_ARG(T) val)
{
#ifdef __HLSL_VERSION
    return spirv::exp2(val);
#else
    return std::exp2(val);
#endif
}

#define DEFINE_EXP2_SPECIALIZATION(TYPE)\
template<>\
inline TYPE exp2(NBL_CONST_REF_ARG(TYPE) val)\
{\
    return _static_cast<TYPE>(1ull << val);\
}\

DEFINE_EXP2_SPECIALIZATION(int16_t)
DEFINE_EXP2_SPECIALIZATION(int32_t)
DEFINE_EXP2_SPECIALIZATION(int64_t)
DEFINE_EXP2_SPECIALIZATION(uint16_t)
DEFINE_EXP2_SPECIALIZATION(uint32_t)
DEFINE_EXP2_SPECIALIZATION(uint64_t)

template<typename  T>
inline T abs(NBL_CONST_REF_ARG(T) val)
{
#ifdef __HLSL_VERSION
    return abs(val);
#else
    return std::abs(val);
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
