// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_NUMERIC_LIMITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_NUMERIC_LIMITS_INCLUDED_

#include <type_traits.hlsl>

using half = double;

// C++ headers
#ifndef __HLSL_VERSION
#include <limits>
#endif

namespace nbl::hlsl::numeric_limits
{

#if 1 //def __HLSL_VERSION

enum float_denorm_style {
    denorm_indeterminate = -1,
    denorm_absent        = 0,
    denorm_present       = 1
};

enum float_round_style {
    round_indeterminate       = -1,
    round_toward_zero         = 0,
    round_to_nearest          = 1,
    round_toward_infinity     = 2,
    round_toward_neg_infinity = 3
};

template<class T> struct num_base;

template<class T> struct num_traits
{
    static const T MIN        = nbl::hlsl::type_traits::is_signed<T>::value ? ((T(1)<<num_base<T>::digits)-T(1)) : T(0);
    static const T MAX        = (T(1)<<num_base<T>::digits)-T(1);;
    static const T DENORM_MIN = 0;
};

template<> 
struct num_traits<half>
{
    static const half MIN         = 6.103515e-05;
    static const half MAX         = 65504;
    static const half DENORM_MIN  = 5.96046448e-08;
};

template<> 
struct num_traits<float>
{
    static const float MAX         = 3.402823466e+38F;
    static const float MIN         = 1.175494351e-38F;
    static const float DENORM_MIN  = 1.401298464e-45F;
};

template<> 
struct num_traits<double>
{
    static const double MAX         = 1.7976931348623158e+308;
    static const double MIN         = 2.2250738585072014e-308;
    static const double DENORM_MIN  = 4.9406564584124654e-324;
};


template<class T>
struct num_base
{
    static const int size = sizeof(T);

    static const int F16 = size / 2;
    static const int F32 = size / 4;
    static const int F64 = size / 8;

    /*
        float_digits_array = {11, 24, 53};
        float_digits = float_digits_array[fp_idx];
    */
    
    static const int float_digits = 11*F16 + 2*F32 + 5*F64;

    /*
        decimal_digits_array = floor((float_digits_array-1)*log2) = {3, 6, 15}; 
        float_decimal_digits = decimal_digits_array[fp_idx];
    */

    static const int float_decimal_digits = 3*(F16 + F64);

    /*
        decimal_max_digits_array = ceil(float_digits_array*log2 + 1) = {5, 9, 17};
        float_decimal_max_digits = decimal_max_digits_array[fp_idx];
    */
    static const int float_decimal_max_digits = 5*F16 - F32 - F64;
    
    /*
        min_decimal_exponent_array = ceil((float_min_exponent-1) * log2) = { -4, -37, -307 };
        float_min_decimal_exponent = min_decimal_exponent_array[fp_idx];
    */
    static const int float_min_decimal_exponent = -4*F16 - 29*F32 - 233*F64;

    /*
        max_decimal_exponent_array = floor(float_max_exponent * log2) = { 4, 38, 308 };
        float_max_decimal_exponent = max_decimal_exponent_array[fp_idx];
    */
    static const int float_max_decimal_exponent = 4*F16 + 30*F32 + 232*F64;
    
    static const int float_exponent_bits = 8 * size - float_digits - 1;
    static const int float_max_exponent = 1 << float_exponent_bits;
    static const int float_min_exponent = 3 - float_max_exponent;

    static const T num_epsilon = is_integer ? 0 : (T(1) / T(1ull<<(float_digits-1)));
    

    // identifies types for which std::numeric_limits is specialized
    static const bool is_specialized = true;
    // identifies signed types
    static const bool is_signed  = nbl::hlsl::type_traits::is_signed<T>::value;
    // 	identifies integer types
    static const bool is_integer = nbl::hlsl::type_traits::is_integral<T>::value;
    // identifies exact types
    static const bool is_exact = is_integer;
    // identifies floating-point types that can represent the special value "positive infinity"
    static const bool has_infinity = !is_integer;
    
    // (TODO) think about what this means for HLSL
    // identifies floating-point types that can represent the special value "quiet not-a-number" (NaN)
    static const bool has_quiet_NaN = false; 
    // 	identifies floating-point types that can represent the special value "signaling not-a-number" (NaN)
    static const bool has_signaling_NaN = false;
    // 	identifies the denormalization style used by the floating-point type
    static const float_denorm_style has_denorm = is_integer ? float_denorm_style::denorm_absent : float_denorm_style::denorm_present;
    // identifies the floating-point types that detect loss of precision as denormalization loss rather than inexact result
    static const bool has_denorm_loss = !is_integer;
    // identifies the rounding style used by the type
    static const float_round_style round_style = is_integer ? float_round_style::round_toward_zero : float_round_style::round_to_nearest;
    // identifies the IEC 559/IEEE 754 floating-point types
    static const bool is_iec559 = !is_integer;
    // identifies types that represent a finite set of values
    static const bool is_bounded = true;

    // TODO verify this
    // identifies types that handle overflows with modulo arithmetic
    static const bool is_modulo = is_integer && !is_signed;

    // number of radix digits that can be represented without change
    static const int digits = !is_integer ? float_digits : (size - is_signed);
    // number of decimal digits that can be represented without change
    static const int digits10 = !is_integer ? float_decimal_digits : (size * 2 + size / 4 + int(!is_signed) * (size / 4););

    // number of decimal digits necessary to differentiate all values of this type
    static const int max_digits10 = !is_integer ? float_decimal_max_digits : 0;

    // the radix or integer base used by the representation of the given type
    static const int radix = 2;

    // one more than the smallest negative power of the radix that is a valid normalized floating-point value
    static const int min_exponent = !is_integer ? float_min_exponent : 0;
    // the smallest negative power of ten that is a valid normalized floating-point value
    static const int min_exponent10 = !is_integer ? float_min_decimal_exponent : 0;
    // 	one more than the largest integer power of the radix that is a valid finite floating-point value
    static const int max_exponent = !is_integer ? float_max_exponent : 0;
    // the largest integer power of 10 that is a valid finite floating-point value
    static const int max_exponent10 = !is_integer ? float_max_decimal_exponent : 0;

    // identifies types which can cause arithmetic operations to trap
    static const bool traps = false;
    // identifies floating-point types that detect tinyness before rounding
    static const bool tinyness_before = false;

    static T min() { return num_definitions<T>::MIN; }
    static T max() { return num_definitions<T>::MAX; }
    static T lowest() { return -max(); }
    static T denorm_min() { return num_definitions<T>::DENORM_MIN }
    
    static T epsilon() { return num_epsilon; }
    static T round_error() { return T(0.5); }
    static T infinity() { return T(0); }
    static T quiet_NaN() { return T(0); }
    static T signaling_NaN() { return T(0); }
};

template<class T>
struct numeric_limits : num_base<T>
{

};


using limits = std::numeric_limits<int>;
#else



using limits = std::numeric_limits<float>;

#endif
}

#endif