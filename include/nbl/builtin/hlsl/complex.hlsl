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


template<typename T>
struct complex_t
{
    T real;
    T imag;
};

template<typename T>
complex_t<T> add(complex_t<T> a, complex_t<T> b)
{
    return a + b;
}

template<typename T>
complex_t<T> subract(complex_t<T> a, complex_t<T> b)
{
    return a - b;
}

template<typename T>
complex_t<T> multiply(complex_t<T> a, complex_t<T> b)
{
    return a * b;
}

template<typename T>
complex_t<T> divide(complex_t<T> a, complex_t<T> b)
{
    return a / b;
}


}
}

#endif

#endif