
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

int identityFunction(in int x)     { return x; }
uint identityFunction(in uint x)   { return x; }
float identityFunction(in float x) { return x; }

// no float version of these because denormalized bit patterns may flush to 0
int and(in int x, in int y)    { return x&y; }
uint and(in uint x, in uint y) { return x&y; }

int xor(in int x, in int y)    { return x^y; }
uint xor(in uint x, in uint y) { return x^y; }

int or(in int x, in int y)    { return x|y; }
uint or(in uint x, in uint y) { return x|y; }


int add(in int x, in int y)       { return x+y; }
uint add(in uint x, in uint y)    { return x+y; }
float add(in float x, in float y) { return x+y; }

int mul(in int x, in int y)       { return x*y; }
uint mul(in uint x, in uint y)    { return x*y; }
float mul(in float x, in float y) { return x*y; }

}
}
}

#endif