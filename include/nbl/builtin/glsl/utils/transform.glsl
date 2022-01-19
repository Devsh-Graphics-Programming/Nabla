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


vec4 nbl_glsl_pseudoMul4x4with3x1(in mat4 m, in vec3 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
}
vec3 nbl_glsl_pseudoMul3x4with3x1(in mat4x3 m, in vec3 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
}
mat4x3 nbl_glsl_pseudoMul4x3with4x3(in mat4x3 lhs, in mat4x3 rhs) // TODO: change name to 3x4with3x4
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

// useful for fast computation of a Normal Matrix (you just need to remember to normalize the transformed normal because of the missing divide by the determinant)
mat3 nbl_glsl_sub3x3TransposeCofactors(in mat3 sub3x3)
{
    return mat3(
        cross(sub3x3[1],sub3x3[2]),
        cross(sub3x3[2],sub3x3[0]),
        cross(sub3x3[0],sub3x3[1])
    );
}
// returns a signflip mask
uint nbl_glsl_sub3x3TransposeCofactors(in mat3 sub3x3, out mat3 sub3x3TransposeCofactors)
{
    sub3x3TransposeCofactors = nbl_glsl_sub3x3TransposeCofactors(sub3x3);
    return floatBitsToUint(dot(sub3x3[0],sub3x3TransposeCofactors[0]))&0x80000000u;
}

// use this if you anticipate flipped/mirrored models
vec3 nbl_glsl_fastNormalTransform(in uint signFlipMask, in mat3 sub3x3TransposeCofactors, in vec3 normal)
{
    vec3 tmp = sub3x3TransposeCofactors*normal;
    const float tmpLenRcp = inversesqrt(dot(tmp,tmp));
    return tmp*uintBitsToFloat(floatBitsToUint(tmpLenRcp)^signFlipMask);
}

//
mat4x3 nbl_glsl_pseudoInverse3x4(in mat4x3 tform)
{
    const mat3 sub3x3Inv = inverse(mat3(tform));
    mat4x3 retval;
    retval[0] = sub3x3Inv[0];
    retval[1] = sub3x3Inv[1];
    retval[2] = sub3x3Inv[2];
    retval[3] = -sub3x3Inv*tform[3];
    return retval;
}

#endif