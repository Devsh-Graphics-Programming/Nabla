// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_TGMATH_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>
// C++ headers
#ifndef __HLSL_VERSION
#include <cmath>
#endif

namespace nbl
{
namespace hlsl
{

namespace impl
{
template<typename T>
struct erf;

template<>
struct erf<float>
{
    static float __call(float _x)
    {
        const float a1 = 0.254829592;
        const float a2 = -0.284496736;
        const float a3 = 1.421413741;
        const float a4 = -1.453152027;
        const float a5 = 1.061405429;
        const float p = 0.3275911;

        float sign = sign(_x);
        float x = abs(_x);
        
        float t = 1.0 / (1.0 + p*x);
        float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
        
        return sign * y;
    }
};


template<typename T>
struct erfInv;

template<>
struct erfInv<float>
{
    static float __call(float _x)
    {
        float x = clamp<float>(_x, -0.99999, 0.99999);
        float w = -log((1.0-x) * (1.0+x));
        float p;
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
            w = sqrt(w) - 3.0;
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
};
}

template<typename T>
T erf(T _x)
{
    return impl::erf<T>::__call(_x);
}

template<typename T>
T erfInv(T _x)
{
    return impl::erfInv<T>::__call(_x);
}


template <typename T>
T rsqrt(T _x)
{
#ifdef __HLSL_VERSION
    return rsqrt(_x);
#else
    return 1.0 / sqrt(_x);
#endif
}

}
}

#endif
