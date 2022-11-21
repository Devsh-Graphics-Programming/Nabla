
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_MATH_TYPELESS_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_TYPELESS_ARITHMETIC_INCLUDED_

// TODO: change header name to binary_operator_functions.glsl


namespace nbl
{
namespace hlsl
{
namespace binops
{

template<typename scaler_t>
scaler_t identityFunction(in scaler_t x)         { return x; }


// no float version of these because denormalized bit patterns may flush to 0
template<typename integer_t>
integer_t and(in integer_t x, in integer_t y)    { return x&y; }

template<typename integer_t>
integer_t xor(in integer_t x, in integer_t y)    { return x^y; }

template<typename integer_t>
integer_t or(in integer_t x, in integer_t y)     { return x|y; }


template<typename scaler_t>
scaler_t add(in scaler_t x, in scaler_t y)       { return x+y; }

template<typename scaler_t>
scaler_t mul(in scaler_t x, in scaler_t y)       { return x*y; }

}
}
}

#endif