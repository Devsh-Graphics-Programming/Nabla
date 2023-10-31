// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_
#define _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_

// C++ headers
#ifndef __HLSL_VERSION
#include <complex>
#endif

namespace nbl
{
namespace hlsl
{



template <typename T>
struct complex_t<vector<T, uint16_t N>>
{
    vector<T, N> real;
    vector<T, N> imag;
};



template <typename T, uint16_t N>
complex_t<vector<T, N>> add(const complex_t<vector<T, N>>& a, const complex_t<vector<T, N>>& b)
{
    complex_t<vector<T, N>> result;
    for (uint16_t i = 0; i < N; ++i)
    {
        result.real[i] = a.real[i] + b.real[i];
        result.imag[i] = a.imag[i] + b.imag[i];
    }

    return result;
}

template <typename T, uint16_t N>
complex_t<vector<T, N>> subtract(const complex_t<vector<T, N>>& a, const complex_t<vector<T, N>>& b)
{
    complex_t<vector<T, N>> result;
    for (uint16_t i = 0; i < N; ++i)
    {
        result.real[i] = a.real[i] - b.real[i];
        result.imag[i] = a.imag[i] - b.imag[i];
    }

    return result;
}

template <typename T, uint16_t N>
complex_t<vector<T, N>> multiply(const complex_t<vector<T, N>>& a, const complex_t<vector<T, N>>& b)
{
    complex_t<vector<T, N>> result;
    for (uint16_t i = 0; i < N; ++i)
    {
        result.real[i] = a.real[i] * b.real[i] - a.imag[i] * b.imag[i];
        result.imag[i] = a.real[i] * b.imag[i] + a.imag[i] * b.real[i];
    }

    return result;
}

template <typename T, uint16_t N>
complex_t<vector<T, N>> divide(const complex_t<vector<T, N>>& a, const complex_t<vector<T, N>>& b)
{
    complex_t<vector<T, N>> result;
    for (uint16_t i = 0; i < N; ++i)
    {
        T denominator = b.real[i] * b.real[i] + b.imag[i] * b.imag[i];
        result.real[i] = (a.real[i] * b.real[i] + a.imag[i] * b.imag[i]) / denominator;
        result.imag[i] = (a.imag[i] * b.real[i] - a.real[i] * b.imag[i]) / denominator;
    }

    return result;
}



}
}