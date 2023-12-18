// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_RADIX_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_RADIX_FFT_INCLUDED_

#include <nbl/builtin/hlsl/complex.hlsl>


namespace nbl
{
namespace hlsl
{

const float64_t PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;

// Fast Fourier Transform
template<typename T, uint32_t N, bool is_inverse>
void fft_radix2(complex_t<T> data[N])
{
    const uint32_t log2N = uint32_t(log2(N));

    // Bit reversal permutation
    for (uint32_t i = 0; i < N; ++i)
    {
        uint32_t j = 0;
        for (uint32_t bit = 0; bit < log2N; ++bit)
            j |= ((i >> bit) & 1) << (log2N - 1 - bit);

        if (j > i)
        {
            complex_t<T> temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }

    // Cooley-Tukey
    for (uint32_t stride = 2; stride <= N; stride *= 2)
    {
        complex_t<T> twiddle_factor;
        if(is_inverse)
        {
            twiddle_factor = { cos(2.0 * PI / stride), sin(2.0 * PI / stride) };
        }
        else
        {
            twiddle_factor = { cos(-2.0 * PI / stride), sin(-2.0 * PI / stride) };
        }
        for (uint32_t start = 0; start < N; start += stride)
        {
            complex_t<T> w = { 1, 0 };
            for (uint32_t i = 0; i < stride / 2; ++i)
            {
                complex_t<T> u = data[start + i];
                complex_t<T> v = mul<T>(w, data[start + i + stride / 2]);
                data[start + i] = plus<T>(u, v);
                data[start + i + stride / 2] = minus<T>(u, v);

                w = mul<T>(w, twiddle_factor);
            }
        }
    }

    // If it's an inverse FFT, divide by N
    if (is_inverse)
    {
        for (uint32_t i = 0; i < N; ++i)
            data[i] = div<T>(data[i], N);
    }
}


}
}

#endif