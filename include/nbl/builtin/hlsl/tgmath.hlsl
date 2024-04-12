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

#ifndef __HLSL_VERSION
    #define NBL_ADDRESS_OF(x) &x
    #define NBL_LANG_SELECT(x, y) x
    #define INTRINSIC_NAMESPACE(x) std::x
#else
    #define INTRINSIC_NAMESPACE(x) x
    #define NBL_ADDRESS_OF(x) x
    #define NBL_LANG_SELECT(x, y) y
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

// TODO numerically better impl of asinh
//template<class T>
//T asinh(T x)
//{
//    return INTRINSIC_NAMESPACE(log)(x + INTRINSIC_NAMESPACE(sqrt)(x*x + T(1)));
//}

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

//template<class T> 
//T expm1(T x) 
//{ 
    // for better implementation, missing function for evaluate_polynomial
    // https://www.boost.org/doc/libs/1_83_0/boost/math/special_functions/expm1.hpp
    //or use std::expm1f, std::expm1l depending on size of T
//}

template<class T> 
T log1p(T x) 
{ 
    // https://www.boost.org/doc/libs/1_83_0/boost/math/special_functions/log1p.hpp
    /*if (x < T(-1))
         "log1p(x) requires x > -1, but got x = %1%.
        */
    if (x == T(-1))
    {
        using uint_type = typename nbl::hlsl::numeric_limits<T>::uint_type;
        return  -nbl::hlsl::numeric_limits<T>::infinity;
    }
    T u = T(1) + x;
    if (u == T(1))
        return x;
    else
        return INTRINSIC_NAMESPACE(log)(u) * (x / (u - T(1)));
}

template<class T>
int32_t ilogb(T x) 
{ 
    using uint_type = typename nbl::hlsl::numeric_limits<T>::uint_type;
    const int32_t shift = (impl::num_base<T>::float_digits-1);
    const uint_type mask = ~(((uint_type(1) << shift) - 1) | (uint_type(1)<<(sizeof(T)*8-1)));
    int32_t bits = (INTRINSIC_NAMESPACE(bit_cast)<uint_type, T>(x) & mask) >> shift; 
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
// NBL_ALIAS_UNARY_FUNCTION(cbrt) // TODO

template<class T>
T hypot(T x, T y)
{
    return INTRINSIC_NAMESPACE(sqrt)(x*x+y*y);
}

// Floating-point manipulation functions 

template<class T>
T copysign(T x, T sign_)
{
    if ((x < 0 && sign_ > 0) || (x > 0 && sign_ < 0))
        return -x;
    return x;
}

// Generate quiet NaN (function)
template<class T>
T nan()
{
    using uint_type = typename nbl::hlsl::numeric_limits<T>::uint_type;
    return INTRINSIC_NAMESPACE(bit_cast)<uint_type, T>(nbl::hlsl::numeric_limits<T>::quiet_NaN);
}
// TODO:
// nextafter	Next representable value (function) 
// nexttoward	Next representable value toward precise value (function)
// erf
// erfc
// tgamma
// lgamma

//template<class T>
//T lgamma(T x)
//{
//    return INTRINSIC_NAMESPACE(log)(INTRINSIC_NAMESPACE(tgamma)(x));
//}

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

// TODO numerically better implementation
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
#undef NBL_ALIAS_FUNCTION_WITH_OUTPUT_PARAM
#undef NBL_ADDRESS_OF
#undef NBL_LANG_SELECT

}
}

#endif

