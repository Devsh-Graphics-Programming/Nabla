
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_COMPLEX_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_COMPLEX_INCLUDED_

#include <nbl/builtin/hlsl/math/constants.hlsl>


namespace nbl
{
namespace hlsl
{
namespace math
{


typedef uint complex16_t;

complex16_t complex16_t_conjugate(in complex16_t complex)
{
    return complex^0x80000000u;
}



template<typename vector_t>
struct complex_t
{
	complex_t<vector_t> expImaginary(in float _theta)
	{
		complex_t<vector_t> result;
		result.real = cos(_theta);
		result.imaginary = sin(_theta);
		return result;
	}

    complex_t<vector_t> operator+(const complex_t<vector_t> other)
    {
    	complex_t<vector_t> result;
    	result.real = real + other.real;
    	result.imaginary = imaginary + other.imaginary;
    	return result;
    }

    complex_t<vector_t> operator-(const complex_t<vector_t> other)
    {
    	complex_t<vector_t> result;
    	result.real = real - other.real;
    	result.imaginary = imaginary - other.imaginary;
    	return result;
    }

    complex_t<vector_t> operator*(const complex_t<vector_t> other)
    {
    	complex_t<vector_t> result;
    	result.real = real * other.real - imaginary * other.imaginary;
    	result.imaginary = real * other.real + imaginary * other.imaginary;
    	return result;
    }

    complex_t<vector_t> conjugate()
    {
    	complex_t<vector_t> result;
    	result.real = real;
    	result.imaginary = -imaginary;
    	return result;
    }
    
    vector_t real, imaginary;
};



namespace fft
{

template<typename scalar_t>
complex_t<scalar_t> twiddle(in uint k, in float N)
{
    complex_t<scalar_t> retval;
    retval.x = cos(-2.f*PI*float(k)/N);
    retval.y = sqrt(1.f-retval.x*retval.x); // twiddle is always half the range, so no conditional -1.f needed
    return retval;
}

template<typename scalar_t>
complex_t<scalar_t> twiddle(in uint k, in uint logTwoN)
{
    return twiddle<scalar_t>(k,float(1<<logTwoN));
}

template<typename scalar_t>
complex_t<scalar_t> twiddle(in bool is_inverse, in uint k, in float N)
{
    complex_t<scalar_t> twiddle = twiddle(k,N);
    if (is_inverse)
        return twiddle.conjugate;
    return twiddle;
}

template<typename scalar_t>
complex_t<scalar_t> twiddle(in bool is_inverse, in uint k, in uint logTwoN)
{
    return twiddle<scalar_t>(is_inverse,k,float(1<<logTwoN));
}



// decimation in time
template<typename scalar_t, typename vector_t>
void DIT_radix2(in complex_t<scalar_t> twiddle, inout complex_t<vector_t> lo, inout complex_t<vector_t> hi)
{
    complex_t<vector_t> wHi = hi * twiddle;
    hi = lo-wHi;
    lo += wHi;
}

// decimation in frequency
template<typename scalar_t, typename vector_t>
void DIF_radix2(in complex_t<scalar_t> twiddle, inout complex_t<vector_t> lo, inout complex_t<vector_t> hi)
{
    complex_t<vector_t> diff = lo-hi;
    lo += hi;
    hi = diff * twiddle;
}


}

// TODO: radices 4,8 and 16
	
}
}
}



#endif