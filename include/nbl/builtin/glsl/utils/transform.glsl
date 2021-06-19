// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_UTILS_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_TRANSFORM_INCLUDED_

mat3 nbl_glsl_mul_with_bounds_wo_gamma(out mat3 error, in mat3 a, in mat3 b, in mat3 b_error)
{
    mat3 retval;
    for (int i=0; i<3; i++)
    {
        vec3 tmp = a[0]*b[i][0];
        retval[i] = tmp;
        error[i] = abs(tmp);
        vec3 additional_error = abs(a[0]*b_error[i][0]);
        for (int j=1; j<3; j++)
        {
            tmp = a[j]*b[i][j];
            retval[i] += tmp;
            error[i] += abs(tmp);
            additional_error += abs(a[j]*b_error[i][j]);
        }
        additional_error *= nbl_glsl_ieee754_rcpgamma(2u)+1.f;
        error[i] += additional_error;
    }
    return retval;
}
mat3 nbl_glsl_mul_with_bounds_wo_gamma(out mat3 error, in mat3 a, in mat3 b)
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
    return retval;
}

mat3 nbl_glsl_mul_with_bounds(out mat3 error, in mat3 a, in mat3 b, in mat3 b_error)
{
    return nbl_glsl_mul_with_bounds_wo_gamma(error,a,b,b_error);
}

mat3 nbl_glsl_mul_with_bounds(out mat3 error, in mat3 a, in mat3 b)
{
    return nbl_glsl_mul_with_bounds_wo_gamma(error,a,b);
}


vec4 nbl_glsl_pseudoMul4x4with3x1(in mat4 m, in vec3 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
}
vec3 nbl_glsl_pseudoMul3x4with3x1(in mat4x3 m, in vec3 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
}
mat4x3 nbl_glsl_pseudoMul4x3with4x3(in mat4x3 lhs, in mat4x3 rhs)
{
    mat4x3 result;
    for (int i = 0; i < 4; i++)
        result[i] = lhs[0] * rhs[i][0] + lhs[1] * rhs[i][1] + lhs[2] * rhs[i][2];
    result[3] += lhs[3];
    return result;
}
mat4 nbl_glsl_pseudoMul4x4with4x3(in mat4 proj, in mat4x3 tform)
{
    mat4 result;
    for (int i = 0; i < 4; i++)
        result[i] = proj[0] * tform[i][0] + proj[1] * tform[i][1] + proj[2] * tform[i][2];
    result[3] += proj[3];
    return result;
}

#endif