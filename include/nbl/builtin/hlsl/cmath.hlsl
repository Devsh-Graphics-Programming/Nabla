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

#define NBL_ALIAS_BINARY_FUNCTION(fn) template<class T> T fn(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y) { return fn(x, y); }
#define NBL_ALIAS_UNARY_FUNCTION2(name,impl)  template<class T> T name(NBL_CONST_REF_ARG(T) x) { return impl(x); }
#define NBL_ALIAS_UNARY_FUNCTION(fn)  NBL_ALIAS_UNARY_FUNCTION2(fn,fn)

#define NBL_ALIAS_FUNCTION_WITH_OUTPUT_PARAM(fn, out_type) \
template<class T>  \
T fn(NBL_CONST_REF_ARG(T) x, NBL_REF_ARG(out_type) y) { \
    T out; \
    T ret = fn(x, out); \
    y = T(out); \
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
NBL_ALIAS_UNARY_FUNCTION(acosh)
NBL_ALIAS_UNARY_FUNCTION(asinh)
NBL_ALIAS_UNARY_FUNCTION(atanh)

// Exponential and logarithmic functions
NBL_ALIAS_UNARY_FUNCTION(exp)
NBL_ALIAS_FUNCTION_WITH_OUTPUT_PARAM(frexp, int32_t)
NBL_ALIAS_FUNCTION_WITH_OUTPUT_PARAM(ldexp, int32_t)
NBL_ALIAS_UNARY_FUNCTION(log)
NBL_ALIAS_UNARY_FUNCTION(log10)
NBL_ALIAS_FUNCTION_WITH_OUTPUT_PARAM(modf, T)
NBL_ALIAS_UNARY_FUNCTION(exp2)
NBL_ALIAS_UNARY_FUNCTION(log2)
NBL_ALIAS_UNARY_FUNCTION(logb,log)

template<class T> 
T expm1(NBL_CONST_REF_ARG(T) x) 
{ 
    return exp(x) - T(1); 
}

template<class T> 
T log1p(NBL_CONST_REF_ARG(T) x) 
{ 
    return log(x + T(1)); 
}

template<class T> 
int ilogb(NBL_CONST_REF_ARG(T) x) 
{ 
    return int(trunc(log(x))); 
}

template<class T> 
T scalbn(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(int32_t) n) 
{ 
    return x * exp2(n); 
}

// Power functions
NBL_ALIAS_BINARY_FUNCTION(pow)
NBL_ALIAS_UNARY_FUNCTION(sqrt)
NBL_ALIAS_UNARY_FUNCTION(cbrt)

template<class T>
T hypot(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) y)
{
    return sqrt(x*x+y*y);
}

// Floating-point manipulation functions

template<class T>
T copysign(NBL_CONST_REF_ARG(T) x, NBL_CONST_REF_ARG(T) sign_)
{
    return T(sign(sign_)) * x;
}

// TODO:
// nan	Generate quiet NaN (function)
// nextafter	Next representable value (function)
// nexttoward	Next representable value toward precise value (function)

// Error and gamma functions

template<class T>
T erf(NBL_CONST_REF_ARG(T) x)
{
    // Bürmann series approximation  
    // https://www.desmos.com/calculator/myf9ylguh1
    // https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
    T E = exp(-x*x);
    T P = T(0.886226925453);
    T re = sqrt(T(1)-E)*(T(1)+T(.155)*E/P*(T(1)-T(.275)*E));
    return copysign(re, x);
}

template<class T>
T erfc(NBL_CONST_REF_ARG(T) x)
{
    return T(1) - erf(x);
}


template<class T>
T tgamma(NBL_CONST_REF_ARG(T) x)
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
        q += p[i] / (z + i - 1);
    }
    
    T t = z + T(6.5);
    return c * sqrt2pi * pow(t, (x-T(.5)))*exp(-t)*q;
}

template<class T>
T lgamma(NBL_CONST_REF_ARG(T) x)
{
    return log(tgamma(x));
}

// Rounding and remainder functions

NBL_ALIAS_UNARY_FUNCTION(ceil)
NBL_ALIAS_UNARY_FUNCTION(floor)
NBL_ALIAS_BINARY_FUNCTION(fmod)
NBL_ALIAS_UNARY_FUNCTION(trunc)
NBL_ALIAS_BINARY_FUNCTION(remainder)
// TODO:
// Below are rounding mode dependent investigate how we handle it
NBL_ALIAS_UNARY_FUNCTION(round)
NBL_ALIAS_UNARY_FUNCTION(rint)
NBL_ALIAS_UNARY_FUNCTION2(nearbyint,round)



// ceil	Round up value (function)
// floor	Round down value (function)
// fmod	Compute remainder of division (function)
// trunc	Truncate value (function)
// round	Round to nearest (function)
// lround	Round to nearest and cast to long integer (function)
// llround	Round to nearest and cast to long long integer (function)
// rint	Round to integral value (function)
// lrint	Round and cast to long integer (function)
// llrint	Round and cast to long long integer (function)
// nearbyint	Round to nearby integral value (function)
// remainder	Compute remainder (IEC 60559) (function)
// remquo	Compute remainder and quotient (function)

// Minimum, maximum, difference functions
// fdim	Positive difference (function)
// fmax	Maximum value (function)
// fmin	Minimum value (function)

// Other functions
// fabs	Compute absolute value (function)
// abs	Compute absolute value (function)
// fma	Multiply-add (function)

}
}

#endif

