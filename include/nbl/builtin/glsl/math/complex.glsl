// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_MATH_COMPLEX_INCLUDED_
#define _NBL_MATH_COMPLEX_INCLUDED_

#include <nbl/builtin/glsl/math/constants.glsl>
#include <nbl/builtin/glsl/math/functions.glsl>

#define nbl_glsl_complex vec2
#define nbl_glsl_cvec2 mat2
#define nbl_glsl_cvec3 mat3x2
#define nbl_glsl_cvec4 mat4x2

nbl_glsl_complex nbl_glsl_expImaginary(in float _theta)
{
    float r = cos(_theta);
    float i = sin(_theta);
    return vec2(r, i);
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

nbl_glsl_complex nbl_glsl_complex_conjugate(in nbl_glsl_complex complex) {
    return complex * vec2(1, -1);
}

#endif
