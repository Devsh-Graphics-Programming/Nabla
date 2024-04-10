// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_LIMITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_LIMITS_INCLUDED_

#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

// C++ headers
#ifndef __HLSL_VERSION
#include <limits>
#include "halfLimits.h"
#endif

/*
    nuumeric_limits C++ API spec elements:
    is_specialized
    is_signed
    is_integer
    is_exact
    has_infinity
    has_quiet_NaN
    has_signaling_NaN
    has_denorm
    has_denorm_loss
    round_style
    is_iec559
    is_bounded
    is_modulo
    digits
    digits10
    max_digits10
    radix
    min_exponent
    min_exponent10
    max_exponent
    max_exponent10
    traps
    tinyness_before
    min
    max
    lowest
    epsilon
    round_error
    infinity
    quiet_NaN
    signaling_NaN
    denorm_min
*/

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

#else

using float_denorm_style = std::float_denorm_style;
using float_round_style  = std::float_round_style;

#endif

// Overlapping definitions

// Below implementations are not intended to be used on C++ side
namespace impl
{


template<class T>
struct num_base : type_identity<T>
{
    NBL_CONSTEXPR_STATIC_INLINE int32_t size = sizeof(T);

    NBL_CONSTEXPR_STATIC_INLINE int32_t S8  = size;
    NBL_CONSTEXPR_STATIC_INLINE int32_t S16 = size / 2;
    NBL_CONSTEXPR_STATIC_INLINE int32_t S32 = size / 4;
    NBL_CONSTEXPR_STATIC_INLINE int32_t S64 = size / 8;

    /*
        float_digits_array = {11, 24, 53};
        float_digits = float_digits_array[fp_idx];
    */
    
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_digits = 11*S16 + 2*S32 + 5*S64;

    /*
        decimal_digits_array = floor((float_digits_array-1)*log2) = {3, 6, 15}; 
        float_decimal_digits = decimal_digits_array[fp_idx];
    */

    NBL_CONSTEXPR_STATIC_INLINE int32_t float_decimal_digits = 3*(S16 + S64);

    /*
        decimal_max_digits_array = ceil(float_digits_array*log2 + 1) = {5, 9, 17};
        float_decimal_max_digits = decimal_max_digits_array[fp_idx];
    */
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_decimal_max_digits = 5*S16 - S32 - S64;
    
    /*
        min_decimal_exponent_array = ceil((float_min_exponent-1) * log2) = { -4, -37, -307 };
        float_min_decimal_exponent = min_decimal_exponent_array[fp_idx];
    */
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_min_decimal_exponent = -4*S16 - 29*S32 - 233*S64;

    /*
        max_decimal_exponent_array = floor(float_max_exponent * log2) = { 4, 38, 308 };
        float_max_decimal_exponent = max_decimal_exponent_array[fp_idx];
    */
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_max_decimal_exponent = 4*S16 + 30*S32 + 232*S64;
    
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_exponent_bits = 8 * size - float_digits - 1;
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_max_exponent = 1l << float_exponent_bits;
    NBL_CONSTEXPR_STATIC_INLINE int32_t float_min_exponent = 3 - float_max_exponent;
    NBL_CONSTEXPR_STATIC_INLINE bool is_bool = is_same<T, bool>::value;

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
    NBL_CONSTEXPR_STATIC_INLINE bool is_modulo = is_integer && !is_signed && !is_bool;

    // number of radix digits that can be represented without change
    NBL_CONSTEXPR_STATIC_INLINE int32_t digits = is_integer ? (is_bool? 1 : 8*size - is_signed) : float_digits;
    // number of decimal digits that can be represented without change
    NBL_CONSTEXPR_STATIC_INLINE int32_t digits10 = is_integer ? (is_bool ? 0 : S8 * 2 + S32 + int32_t(!is_signed) * S64) : float_decimal_digits;

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
    
};

template<class T> 
struct num_traits : num_base<T>
{
    // have to be weird like that to avoid a warning
    NBL_CONSTEXPR_STATIC_INLINE T min            = T(num_base<T>::is_signed)<<(num_base<T>::is_signed ? num_base<T>::digits:0);
    // FIXME: Lots of warnings with `T=bool`
    NBL_CONSTEXPR_STATIC_INLINE T max            = ~min;
    NBL_CONSTEXPR_STATIC_INLINE T denorm_min     = T(0);
    NBL_CONSTEXPR_STATIC_INLINE T quiet_NaN      = T(0);
    NBL_CONSTEXPR_STATIC_INLINE T signaling_NaN  = T(0);
    NBL_CONSTEXPR_STATIC_INLINE T infinity       = T(0);
};

template<> 
struct num_traits<float16_t> : num_base<float16_t>
{
    // since float16_t has no constexpr ctor we have to use float32_t constants on cpp side which is only needed for testing
    using type = 
#ifdef __cplusplus
    float32_t
#else
    float16_t
#endif
;
    NBL_CONSTEXPR_STATIC_INLINE type      min           = 6.103515e-05F;
    NBL_CONSTEXPR_STATIC_INLINE type      max           = 65504.0F;
    NBL_CONSTEXPR_STATIC_INLINE type      denorm_min    = 5.96046448e-08F;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t  quiet_NaN     = 0x7FFF;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t  signaling_NaN = 0x7DFF;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t  infinity      = 0x7C00;
};

template<> 
struct num_traits<float32_t> : num_base<float32_t>
{
    NBL_CONSTEXPR_STATIC_INLINE float32_t max           = 3.402823466e+38F;
    NBL_CONSTEXPR_STATIC_INLINE float32_t min           = 1.175494351e-38F;
    NBL_CONSTEXPR_STATIC_INLINE float32_t denorm_min    = 1.401298464e-45F;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t  quiet_NaN     = 0x7FC00000u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t  signaling_NaN = 0x7FC00001u;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t  infinity      = 0x7F800000u;
};

template<> 
struct num_traits<float64_t> : num_base<float64_t>
{
    NBL_CONSTEXPR_STATIC_INLINE float64_t max           = 1.7976931348623158e+308;
    NBL_CONSTEXPR_STATIC_INLINE float64_t min           = 2.2250738585072014e-308;
    NBL_CONSTEXPR_STATIC_INLINE float64_t denorm_min    = 4.9406564584124654e-324;
    NBL_CONSTEXPR_STATIC_INLINE uint64_t  quiet_NaN     = 0x7FF8000000000000ull;
    NBL_CONSTEXPR_STATIC_INLINE uint64_t  signaling_NaN = 0x7FF0000000000001ull;
    NBL_CONSTEXPR_STATIC_INLINE uint64_t  infinity      = 0x7FF0000000000000ull;
};

template<class T>
struct numeric_limits : num_traits<T>
{
    using type = typename num_traits<T>::type;
    NBL_CONSTEXPR_STATIC_INLINE type lowest  = num_traits<T>::is_integer ? num_traits<T>::min : -num_traits<T>::max;
    // FIXME: warning C4293 `<<`: shift count negative or too big (only when instantiating with `int8_t`
    NBL_CONSTEXPR_STATIC_INLINE type epsilon = num_traits<T>::is_integer ? type(0) : (type(1) / type(1ull<<(num_traits<T>::float_digits-1)));
    NBL_CONSTEXPR_STATIC_INLINE type round_error = type(num_traits<T>::is_iec559)/type(2.0);
};

}

#ifdef __HLSL_VERSION

template<class T>
struct numeric_limits : impl::numeric_limits<T> {};

#else


template<class T>
struct numeric_limits : std::numeric_limits<T>
{
    using base = std::numeric_limits<T>;
    using uint_type = std::remove_cvref_t<decltype(impl::num_traits<T>::infinity)>;

    NBL_CONSTEXPR_STATIC_INLINE T min = base::min();
    NBL_CONSTEXPR_STATIC_INLINE T max = base::max();
    NBL_CONSTEXPR_STATIC_INLINE T lowest = base::lowest();
    NBL_CONSTEXPR_STATIC_INLINE T denorm_min = base::denorm_min();
    NBL_CONSTEXPR_STATIC_INLINE T epsilon = base::epsilon();
    NBL_CONSTEXPR_STATIC_INLINE T round_error = base::round_error();

    NBL_CONSTEXPR_STATIC_INLINE uint_type quiet_NaN     = std::bit_cast<uint_type>(base::quiet_NaN());
    NBL_CONSTEXPR_STATIC_INLINE uint_type signaling_NaN = std::bit_cast<uint_type>(base::signaling_NaN());
    NBL_CONSTEXPR_STATIC_INLINE uint_type infinity      = std::bit_cast<uint_type>(base::infinity());
};

#endif
}
}

#endif

