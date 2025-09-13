// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_UTILS_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_TRANSFORM_INCLUDED_

#include <nbl/builtin/glsl/limits/numeric.glsl>

// move to ieee754 header?
mat3 nbl_glsl_mul_with_bounds_wo_gamma(out mat3 error, in mat3 a, in mat3 b, in float b_relative_error)
{
    mat3 retval;
    for (int i=0; i<3; i++)
    {
        vec3 tmp = a[0]*b[i][0];
        retval[i] = tmp;
        error[i] = abs(tmp);
        for (int j=1; j<3; j++)
        {
            tmp = a[j]*b[i][j];
            retval[i] += tmp;
            error[i] += abs(tmp);
        }
    }
    const float error_factor = 1.f+b_relative_error/nbl_glsl_numeric_limits_float_epsilon(2);
    error *= error_factor;
    return retval;
}
mat3 nbl_glsl_mul_with_bounds(out mat3 error, in mat3 a, in mat3 b, in float b_relative_error)
{
    mat3 retval = nbl_glsl_mul_with_bounds_wo_gamma(error,a,b,b_relative_error);
    error *= nbl_glsl_ieee754_gamma(2u);
    return retval;
}

#endif