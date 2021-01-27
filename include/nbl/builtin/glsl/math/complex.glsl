// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_MATH_COMPLEX_INCLUDED_
#define _NBL_MATH_COMPLEX_INCLUDED_

#include <nbl/builtin/glsl/math/constants.glsl>
#include <nbl/builtin/glsl/math/functions.glsl>

vec2 nbl_glsl_eITheta(in float _theta)
{
    // Use sincos from math/functions.glsl?
    float r = cos(_theta);
    float i = sin(_theta);
    return vec2(r, i);
}

vec2 nbl_glsl_complex_mul(in vec2 rhs, in vec2 lhs)
{
    float r = rhs.x * lhs.x - rhs.y * lhs.y;
    float i = rhs.x * lhs.y + rhs.y * lhs.x;
    return vec2(r, i);
}

vec2 nbl_glsl_complex_add(in vec2 rhs, in vec2 lhs)
{
    return rhs + lhs;
}

vec2 nbl_glsl_complex_conjugate(in vec2 complex) {
    return complex * vec2(1, -1);
}

#endif
