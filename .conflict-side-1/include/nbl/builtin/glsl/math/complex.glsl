// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_MATH_COMPLEX_INCLUDED_
#define _NBL_MATH_COMPLEX_INCLUDED_

#include <nbl/builtin/glsl/math/constants.glsl>
#include <nbl/builtin/glsl/math/functions.glsl>

#define nbl_glsl_complex16_t uint

#define nbl_glsl_complex vec2
#define nbl_glsl_cvec2 mat2
#define nbl_glsl_cvec3 mat3x2
#define nbl_glsl_cvec4 mat4x2

nbl_glsl_complex nbl_glsl_expImaginary(in float _theta)
{
    return vec2(cos(_theta),sin(_theta));
}

nbl_glsl_complex nbl_glsl_complex_mul(in nbl_glsl_complex rhs, in nbl_glsl_complex lhs)
{
    float r = rhs.x * lhs.x - rhs.y * lhs.y;
    float i = rhs.x * lhs.y + rhs.y * lhs.x;
    return vec2(r, i);
}

nbl_glsl_complex nbl_glsl_complex_add(in nbl_glsl_complex rhs, in nbl_glsl_complex lhs)
{
    return rhs + lhs;
}

nbl_glsl_complex16_t nbl_glsl_complex16_t_conjugate(in nbl_glsl_complex16_t complex) {
    return complex^0x80000000u;
}
nbl_glsl_complex nbl_glsl_complex_conjugate(in nbl_glsl_complex complex) {
    return nbl_glsl_complex(complex.x,-complex.y);
}


// FFT
nbl_glsl_complex nbl_glsl_FFT_twiddle(in uint k, in float N)
{
    nbl_glsl_complex retval;
    retval.x = cos(-2.f*nbl_glsl_PI*float(k)/N);
    retval.y = sqrt(1.f-retval.x*retval.x); // twiddle is always half the range, so no conditional -1.f needed
    return retval;
}
nbl_glsl_complex nbl_glsl_FFT_twiddle(in uint k, in uint logTwoN)
{
    return nbl_glsl_FFT_twiddle(k,float(1<<logTwoN));
}

nbl_glsl_complex nbl_glsl_FFT_twiddle(in bool is_inverse, in uint k, in float N)
{
    nbl_glsl_complex twiddle = nbl_glsl_FFT_twiddle(k,N);
    if (is_inverse)
        return nbl_glsl_complex_conjugate(twiddle);
    return twiddle;
}
nbl_glsl_complex nbl_glsl_FFT_twiddle(in bool is_inverse, in uint k, in uint logTwoN)
{
    return nbl_glsl_FFT_twiddle(is_inverse,k,float(1<<logTwoN));
}



// decimation in time
void nbl_glsl_FFT_DIT_radix2(in nbl_glsl_complex twiddle, inout nbl_glsl_complex lo, inout nbl_glsl_complex hi)
{
    nbl_glsl_complex wHi = nbl_glsl_complex_mul(hi,twiddle);
    hi = lo-wHi;
    lo += wHi;
}

// decimation in frequency
void nbl_glsl_FFT_DIF_radix2(in nbl_glsl_complex twiddle, inout nbl_glsl_complex lo, inout nbl_glsl_complex hi)
{
    nbl_glsl_complex diff = lo-hi;
    lo += hi;
    hi = nbl_glsl_complex_mul(diff,twiddle);
}

// TODO: radices 4,8 and 16

#endif
