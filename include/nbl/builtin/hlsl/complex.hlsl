// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_
#define _NBL_BUILTIN_HLSL_COMPLEX_INCLUDED_



#ifdef __HLSL_VERSION


namespace nbl
{
namespace hlsl
{
namespace complex
{


template<typename T>
struct complex_t
{
    T real;
    T imag;
};

template<typename T>
complex_t<T> plus(const complex_t<T> a, const complex_t<T> b)
{
    complex_t<T> result;

    result.real = a.real + b.real;
    result.imag = a.imag + b.imag;

    return result;
}

template<typename T>
complex_t<T> minus(const complex_t<T> a, const complex_t<T> b)
{
    complex_t<T> result;

    result.real = a.real - b.real;
    result.imag = a.imag - b.imag;

    return result;
}

template<typename T>
complex_t<T> mul(const complex_t<T> a, const complex_t<T> b)
{
    complex_t<T> result;

    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;

    return result;
}

// multiply by scalar
template<typename T>
complex_t<T> mul(const complex_t<T> c, T scalar)
{
    complex_t<T> result;
    result.real = c.real * scalar;
    result.imag = c.imag * scalar;

    return result;
}


template<typename T>
complex_t<T> div(const complex_t<T> a, const complex_t<T> b)
{
    complex_t<T> result;

    T denominator = b.real * b.real + b.imag * b.imag;
    result.real = (a.real * b.real + a.imag * b.imag) / denominator;
    result.imag = (a.imag * b.real - a.real * b.imag) / denominator;

    return result;
}

// Divide by scalar
template<typename T>
complex_t<T> div(const complex_t<T> c, T scalar)
{
    complex_t<T> result;
    result.real = c.real / scalar;
    result.imag = c.imag / scalar;

    return result;
}


}
}
}


#endif

#endif