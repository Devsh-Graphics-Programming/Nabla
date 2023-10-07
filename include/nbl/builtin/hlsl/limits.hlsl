// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_LIMITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_LIMITS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>

// C++ headers
#ifndef __HLSL_VERSION
#include <limits>
#endif

namespace nbl
{
namespace hlsl
{

#ifdef __HLSL_VERSION

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

template<class T>
struct numeric_limits;

template<class T> 
struct num_traits
{
    NBL_CONSTEXPR_STATIC_INLINE T MIN            = numeric_limits<T>::is_signed ? (T(1)<<numeric_limits<T>::digits) : T(0);
    NBL_CONSTEXPR_STATIC_INLINE T MAX            = ~(T(numeric_limits<T>::is_signed)<<numeric_limits<T>::digits);
    NBL_CONSTEXPR_STATIC_INLINE T DENORM_MIN     = T(0);
    NBL_CONSTEXPR_STATIC_INLINE T QUIET_NAN      = T(0);
    NBL_CONSTEXPR_STATIC_INLINE T SIGNALING_NAN  = T(0);
};

#ifdef NBL_HALF_DEFINED
template<> 
struct num_traits<half>
{
    NBL_CONSTEXPR_STATIC_INLINE half MIN         = 6.103515e-05;
    NBL_CONSTEXPR_STATIC_INLINE half MAX         = 65504;
    NBL_CONSTEXPR_STATIC_INLINE half DENORM_MIN  = 5.96046448e-08;
};
#endif

template<> 
struct num_traits<float32_t>
{
    NBL_CONSTEXPR_STATIC_INLINE float32_t MAX           = 3.402823466e+38F;
    NBL_CONSTEXPR_STATIC_INLINE float32_t MIN           = 1.175494351e-38F;
    NBL_CONSTEXPR_STATIC_INLINE float32_t DENORM_MIN    = 1.401298464e-45F;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t  QUIET_NAN     = 0x7FC00000;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t  SIGNALING_NAN = 0x7FC00001;
};

template<> 
struct num_traits<float64_t>
{
    NBL_CONSTEXPR_STATIC_INLINE float64_t MAX           = 1.7976931348623158e+308;
    NBL_CONSTEXPR_STATIC_INLINE float64_t MIN           = 2.2250738585072014e-308;
    NBL_CONSTEXPR_STATIC_INLINE float64_t DENORM_MIN    = 4.9406564584124654e-324;
    NBL_CONSTEXPR_STATIC_INLINE uint64_t  QUIET_NAN     = 0x7FF8000000000000ull;
    NBL_CONSTEXPR_STATIC_INLINE uint64_t  SIGNALING_NAN = 0x7FF0000000000001ull;
};

template<class T>
struct numeric_limits
{
    NBL_CONSTEXPR_STATIC_INLINE int32_t size = sizeof(T);

    NBL_CONSTEXPR_STATIC_INLINE int32_t F16 = size / 2;
    NBL_CONSTEXPR_STATIC_INLINE int32_t F32 = size / 4;
    NBL_CONSTEXPR_STATIC_INLINE int32_t F64 = size / 8;

    /*
        float_digits_array = {11, 24, 53};
        float_digits = float_digits_array[fp_idx];
    */
    
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_digits = 11*F16 + 2*F32 + 5*F64;

    /*
        decimal_digits_array = floor((float_digits_array-1)*log2) = {3, 6, 15}; 
        float_decimal_digits = decimal_digits_array[fp_idx];
    */

    NBL_CONSTEXPR_STATIC_INLINE int32_t float_decimal_digits = 3*(F16 + F64);

    /*
        decimal_max_digits_array = ceil(float_digits_array*log2 + 1) = {5, 9, 17};
        float_decimal_max_digits = decimal_max_digits_array[fp_idx];
    */
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_decimal_max_digits = 5*F16 - F32 - F64;
    
    /*
        min_decimal_exponent_array = ceil((float_min_exponent-1) * log2) = { -4, -37, -307 };
        float_min_decimal_exponent = min_decimal_exponent_array[fp_idx];
    */
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_min_decimal_exponent = -4*F16 - 29*F32 - 233*F64;

    /*
        max_decimal_exponent_array = floor(float_max_exponent * log2) = { 4, 38, 308 };
        float_max_decimal_exponent = max_decimal_exponent_array[fp_idx];
    */
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_max_decimal_exponent = 4*F16 + 30*F32 + 232*F64;
    
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_exponent_bits = 8 * size - float_digits - 1;
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_max_exponent = 1 << float_exponent_bits;
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_min_exponent = 3 - float_max_exponent;


    // identifies types for which std::numeric_limits is specialized
    NBL_CONSTEXPR_STATIC_INLINE bool is_specialized = true;
    // identifies signed types
    NBL_CONSTEXPR_STATIC_INLINE bool is_signed  = nbl::hlsl::is_signed<T>::value;
    // 	identifies integer types
    NBL_CONSTEXPR_STATIC_INLINE bool is_integer = is_integral<T>::value;
    // identifies exact types
    NBL_CONSTEXPR_STATIC_INLINE bool is_exact = is_integer;
    // identifies floating-point types that can represent the special value "positive infinity"
    NBL_CONSTEXPR_STATIC_INLINE bool has_infinity = !is_integer;
    
    NBL_CONSTEXPR_STATIC_INLINE T num_epsilon = is_integer ? 0 : (T(1) / T(1ull<<(float_digits-1)));
    
    // (TODO) think about what this means for HLSL
    // identifies floating-point types that can represent the special value "quiet not-a-number" (NaN)
    NBL_CONSTEXPR_STATIC_INLINE bool has_quiet_NaN = !is_integer; 
    // 	identifies floating-point types that can represent the special value "signaling not-a-number" (NaN)
    NBL_CONSTEXPR_STATIC_INLINE bool has_signaling_NaN = !is_integer;
    // 	identifies the denormalization style used by the floating-point type
    NBL_CONSTEXPR_STATIC_INLINE float_denorm_style has_denorm = is_integer ? float_denorm_style::denorm_absent : float_denorm_style::denorm_present;
    // identifies the floating-point types that detect loss of precision as denormalization loss rather than inexact result
    NBL_CONSTEXPR_STATIC_INLINE bool has_denorm_loss = false;
    // identifies the rounding style used by the type
    NBL_CONSTEXPR_STATIC_INLINE float_round_style round_style = is_integer ? float_round_style::round_toward_zero : float_round_style::round_to_nearest;
    // identifies the IEC 559/IEEE 754 floating-point types
    NBL_CONSTEXPR_STATIC_INLINE bool is_iec559 = !is_integer;
    // identifies types that represent a finite set of values
    NBL_CONSTEXPR_STATIC_INLINE bool is_bounded = true;

    // TODO verify this
    // identifies types that handle overflows with modulo arithmetic
    NBL_CONSTEXPR_STATIC_INLINE bool is_modulo = is_integer && !is_signed;

    // number of radix digits that can be represented without change
    NBL_CONSTEXPR_STATIC_INLINE int32_t digits = !is_integer ? float_digits : (8*size - is_signed);
    // number of decimal digits that can be represented without change
    NBL_CONSTEXPR_STATIC_INLINE int32_t digits10 = !is_integer ? float_decimal_digits : (size * 2 + size / 4 + int32_t(!is_signed) * (size / 8));

    // number of decimal digits necessary to differentiate all values of this type
    NBL_CONSTEXPR_STATIC_INLINE int32_t max_digits10 = !is_integer ? float_decimal_max_digits : 0;

    // the radix or integer base used by the representation of the given type
    NBL_CONSTEXPR_STATIC_INLINE int32_t radix = 2;

    // one more than the smallest negative power of the radix that is a valid normalized floating-point value
    NBL_CONSTEXPR_STATIC_INLINE int32_t min_exponent = !is_integer ? float_min_exponent : 0;
    // the smallest negative power of ten that is a valid normalized floating-point value
    NBL_CONSTEXPR_STATIC_INLINE int32_t min_exponent10 = !is_integer ? float_min_decimal_exponent : 0;
    // 	one more than the largest integer power of the radix that is a valid finite floating-point value
    NBL_CONSTEXPR_STATIC_INLINE int32_t max_exponent = !is_integer ? float_max_exponent : 0;
    // the largest integer power of 10 that is a valid finite floating-point value
    NBL_CONSTEXPR_STATIC_INLINE int32_t max_exponent10 = !is_integer ? float_max_decimal_exponent : 0;

    // identifies types which can cause arithmetic operations to trap
    NBL_CONSTEXPR_STATIC_INLINE bool traps = false;
    // identifies floating-point types that detect tinyness before rounding
    NBL_CONSTEXPR_STATIC_INLINE bool tinyness_before = false;

    static T min() { return num_traits<T>::MIN; }
    static T max() { return num_traits<T>::MAX; }
    static T lowest() { return is_integer ? min() : -max(); }
    static T denorm_min() { return num_traits<T>::DENORM_MIN; }
    
    static T epsilon() { return num_epsilon; }
    static T round_error() { return T(0.5); }
    static T infinity() { return T(1e+300 * 1e+300); }
    static T quiet_NaN() { return bit_cast<T>(num_traits<T>::QUIET_NAN); }
    static T signaling_NaN() { return bit_cast<T>(num_traits<T>::SIGNALING_NAN); }
};


#else

template<class T>
using numeric_limits = std::numeric_limits<T>;

#endif
}
}

#endif

