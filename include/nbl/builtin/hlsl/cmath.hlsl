// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_CMATH_INCLUDED_
#define _NBL_BUILTIN_HLSL_CMATH_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>

// C++ headers
#ifndef __HLSL_VERSION
#include <cmath>
#endif

namespace nbl
{
namespace hlsl
{


#ifdef __cplusplus
#define INTRINSIC_NAMESPACE(x) std::x
#else
#define INTRINSIC_NAMESPACE(x) x
#endif

#define NBL_ALIAS_BINARY_FUNCTION2(name,impl) template<class T> T name(T x, T y) { return INTRINSIC_NAMESPACE(impl)(x, y); }
#define NBL_ALIAS_BINARY_FUNCTION(fn) NBL_ALIAS_BINARY_FUNCTION2(fn,fn)
#define NBL_ALIAS_UNARY_FUNCTION2(name,impl)  template<class T> T name(T x) { return INTRINSIC_NAMESPACE(impl)(x); }
#define NBL_ALIAS_UNARY_FUNCTION(fn)  NBL_ALIAS_UNARY_FUNCTION2(fn,fn)

#define NBL_ALIAS_FUNCTION_WITH_OUTPUT_PARAM(fn, out_type) \
template<class T>  \
T fn(T x, NBL_REF_ARG(out_type) y) { \
    NBL_LANG_SELECT(out_type, T) out_; \
    T ret = INTRINSIC_NAMESPACE(fn)(x, NBL_ADDRESS_OF(out_)); \
    y = out_type(out_); \
    return ret; \
}

// Trigonometric functions
NBL_ALIAS_UNARY_FUNCTION(cos)
NBL_ALIAS_UNARY_FUNCTION(sin)
NBL_ALIAS_UNARY_FUNCTION(tan)
NBL_ALIAS_UNARY_FUNCTION(acos)
NBL_ALIAS_UNARY_FUNCTION(asin)
NBL_ALIAS_UNARY_FUNCTION(atan)
NBL_ALIAS_BINARY_FUNCTION(atan2)

// Hyperbolic functions
NBL_ALIAS_UNARY_FUNCTION(cosh)
NBL_ALIAS_UNARY_FUNCTION(sinh)
NBL_ALIAS_UNARY_FUNCTION(tanh)

template<class T>
T acosh(T x)
{
    return INTRINSIC_NAMESPACE(log)(x + INTRINSIC_NAMESPACE(sqrt)(x*x - T(1)));
}

template<class T>
T asinh(T x)
{
    return INTRINSIC_NAMESPACE(log)(x + INTRINSIC_NAMESPACE(sqrt)(x*x + T(1)));
}

template<class T>
T atanh(T x)
{
    return T(0.5) * INTRINSIC_NAMESPACE(log)((T(1)+x)/(T(1)-x));
}


// Exponential and logarithmic functions
NBL_ALIAS_UNARY_FUNCTION(exp)
NBL_ALIAS_FUNCTION_WITH_OUTPUT_PARAM(frexp, int32_t)
NBL_ALIAS_BINARY_FUNCTION(ldexp)
NBL_ALIAS_UNARY_FUNCTION(log)
NBL_ALIAS_UNARY_FUNCTION(log10)
NBL_ALIAS_FUNCTION_WITH_OUTPUT_PARAM(modf, T)
NBL_ALIAS_UNARY_FUNCTION(exp2)
NBL_ALIAS_UNARY_FUNCTION(log2)
NBL_ALIAS_UNARY_FUNCTION2(logb,log)

template<class T> 
T expm1(T x) 
{ 
    return INTRINSIC_NAMESPACE(exp)(x) - T(1); 
}

template<class T> 
T log1p(T x) 
{ 
    return INTRINSIC_NAMESPACE(log)(x + T(1)); 
}

template<class T>
int32_t ilogb(T x) 
{ 
    using uint_type = typename nbl::hlsl::numeric_limits<T>::uint_type;
    const int32_t shift = (impl::num_base<T>::float_digits-1);
    const uint_type mask = ~(((uint_type(1) << shift) - 1) | (uint_type(1)<<(sizeof(T)*8-1)));
    int32_t bits = (bit_cast<uint_type, T>(x) & mask) >> shift;
    return bits + impl::num_base<T>::min_exponent - 2;
}

template<class T> 
T scalbn(T x, int32_t n) 
{ 
    return x * INTRINSIC_NAMESPACE(exp2)(n); 
}

// Power functions
NBL_ALIAS_BINARY_FUNCTION(pow)
NBL_ALIAS_UNARY_FUNCTION(sqrt)
NBL_ALIAS_UNARY_FUNCTION(cbrt)

template<class T>
T hypot(T x, T y)
{
    return INTRINSIC_NAMESPACE(sqrt)(x*x+y*y);
}

// Floating-point manipulation functions

template<class T>
T copysign(T x, T sign_)
{
    return sign_ < T(0) ? -x : x;
}

// TODO:
// nan	Generate quiet NaN (function)
// nextafter	Next representable value (function)
// nexttoward	Next representable value toward precise value (function)

// Error and gamma functions

template<class T>
T erf(T x)
{
    // Bürmann series approximation  
    // https://www.desmos.com/calculator/myf9ylguh1
    // https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
    T E = INTRINSIC_NAMESPACE(exp)(-x*x);
    T P = T(0.886226925453);
    T re = INTRINSIC_NAMESPACE(sqrt)(T(1)-E)*(T(1)+T(.155)*E/P*(T(1)-T(.275)*E));
    return INTRINSIC_NAMESPACE(copysign)(re, x);
}

template<class T>
T erfc(T x)
{
    return T(1) - erf(x);
}


template<class T>
T tgamma(T x)
{
    // TODO:
    // Investigate this approximation since margin of error seems to be high
    // https://www.desmos.com/calculator/ivfbrvxha8
    // https://en.wikipedia.org/wiki/Lanczos_approximation
    T pi = T(3.14159265358979323846);
    T sqrt2pi = T(2.50662827463);
    T p[] = { 
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };

    T c = T(1);
    if (x < T(0.5))
    {
        c = pi / (sin(pi * x));
        x = T(1)-x;
    }

    T q = p[0];
    for(uint32_t i = 1; i < sizeof(p)/sizeof(p[0]); ++i)
    {
        q += p[i] / (x + i - 1);
    }
    
    T t = x + T(6.5);
    return c * sqrt2pi * INTRINSIC_NAMESPACE(pow)(t, (x-T(.5)))*INTRINSIC_NAMESPACE(exp)(-t)*q;
}

template<class T>
T lgamma(T x)
{
    return INTRINSIC_NAMESPACE(log)(INTRINSIC_NAMESPACE(tgamma)(x));
}

// Rounding and remainder functions

NBL_ALIAS_UNARY_FUNCTION(ceil)
NBL_ALIAS_UNARY_FUNCTION(floor)
NBL_ALIAS_BINARY_FUNCTION(fmod)
NBL_ALIAS_UNARY_FUNCTION(trunc)

// TODO:
// Below are rounding mode dependent investigate how we handle it
NBL_ALIAS_UNARY_FUNCTION(round)
NBL_ALIAS_UNARY_FUNCTION(rint)
NBL_ALIAS_UNARY_FUNCTION2(nearbyint,round)

template<class T>
T remquo(T num, T denom, NBL_REF_ARG(int32_t) quot)
{
    quot = int32_t(INTRINSIC_NAMESPACE(round)(num / denom));
    return num - quot * denom;
}

template<class T>
T remainder(T num, T denom)
{
    int32_t q;
    return remquo(num, denom, q);
}

// Other functions
NBL_ALIAS_UNARY_FUNCTION(abs)
NBL_ALIAS_UNARY_FUNCTION2(fabs, abs)
template<class T>
T fma(T a, T b, T c)
{
    return INTRINSIC_NAMESPACE(fma)(a,b,c);
}

// Minimum, maximum, difference functions

NBL_ALIAS_BINARY_FUNCTION2(fmax, max)
NBL_ALIAS_BINARY_FUNCTION2(fmin, min)
template<class T>
T fdim(T x, T y)
{
    return INTRINSIC_NAMESPACE(max)(T(0),x-y);
}

#undef NBL_ALIAS_BINARY_FUNCTION2
#undef NBL_ALIAS_BINARY_FUNCTION
#undef NBL_ALIAS_UNARY_FUNCTION2
#undef NBL_ALIAS_UNARY_FUNCTION
#undef INTRINSIC_NAMESPACE


}
}

#endif

