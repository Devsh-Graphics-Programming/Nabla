
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_COMPLEX_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_COMPLEX_INCLUDED_

#include <nbl/builtin/hlsl/math/constants.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>



namespace nbl
{
namespace hlsl
{
namespace math
{
namespace complex
{


#define complex16_t uint

#define complex float2
#define cfloat2 float2x2
#define cfloat3 mat2x3
#define cfloat4 mat2x4


complex expImaginary(in float _theta)
{
    return float2(cos(_theta),sin(_theta));
}

complex complex_mul(in complex rhs, in complex lhs)
{
    float r = rhs.x * lhs.x - rhs.y * lhs.y;
    float i = rhs.x * lhs.y + rhs.y * lhs.x;
    return float2(r, i);
}

complex complex_add(in complex rhs, in complex lhs)
{
    return rhs + lhs;
}

complex16_t complex16_t_conjugate(in complex16_t complex) {
    return complex^0x80000000u;
}
complex complex_conjugate(in complex complex) {
    return complex(complex.x,-complex.y);
}


// FFT
complex FFT_twiddle(in uint k, in float N)
{
    complex retval;
    retval.x = cos(-2.f*PI*float(k)/N);
    retval.y = sqrt(1.f-retval.x*retval.x); // twiddle is always half the range, so no conditional -1.f needed
    return retval;
}
complex FFT_twiddle(in uint k, in uint logTwoN)
{
    return FFT_twiddle(k,float(1<<logTwoN));
}

complex FFT_twiddle(in bool is_inverse, in uint k, in float N)
{
    complex twiddle = FFT_twiddle(k,N);
    if (is_inverse)
        return complex_conjugate(twiddle);
    return twiddle;
}
complex FFT_twiddle(in bool is_inverse, in uint k, in uint logTwoN)
{
    return FFT_twiddle(is_inverse,k,float(1<<logTwoN));
}



// decimation in time
void FFT_DIT_radix2(in complex twiddle, inout complex lo, inout complex hi)
{
    complex wHi = complex_mul(hi,twiddle);
    hi = lo-wHi;
    lo += wHi;
}

// decimation in frequency
void FFT_DIF_radix2(in complex twiddle, inout complex lo, inout complex hi)
{
    complex diff = lo-hi;
    lo += hi;
    hi = complex_mul(diff,twiddle);
}

// TODO: radices 4,8 and 16

	
}
}
}
}



#endif