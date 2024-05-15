// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_LINALG_FAST_AFFINE_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_LINALG_FAST_AFFINE_INCLUDED_

#include <nbl/builtin/hlsl/concepts.hlsl>

#if 0 // TODO
vec4 pseudoMul4x4with3x1(in mat4 m, in vec3 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
}
vec3 pseudoMul3x4with3x1(in mat4x3 m, in vec3 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
}
mat4x3 pseudoMul4x3with4x3(in mat4x3 lhs, in mat4x3 rhs) // TODO: change name to 3x4with3x4
{
    mat4x3 result;
    for (int i = 0; i < 4; i++)
        result[i] = lhs[0] * rhs[i][0] + lhs[1] * rhs[i][1] + lhs[2] * rhs[i][2];
    result[3] += lhs[3];
    return result;
}
mat4 pseudoMul4x4with4x3(in mat4 proj, in mat4x3 tform)
{
    mat4 result;
    for (int i = 0; i < 4; i++)
        result[i] = proj[0] * tform[i][0] + proj[1] * tform[i][1] + proj[2] * tform[i][2];
    result[3] += proj[3];
    return result;
}

// useful for fast computation of a Normal Matrix (you just need to remember to normalize the transformed normal because of the missing divide by the determinant)
mat3 sub3x3TransposeCofactors(in mat3 sub3x3)
{
    return mat3(
        cross(sub3x3[1],sub3x3[2]),
        cross(sub3x3[2],sub3x3[0]),
        cross(sub3x3[0],sub3x3[1])
    );
}
// returns a signflip mask
uint sub3x3TransposeCofactors(in mat3 sub3x3, out mat3 sub3x3TransposeCofactors)
{
    sub3x3TransposeCofactors = sub3x3TransposeCofactors(sub3x3);
    return floatBitsToUint(dot(sub3x3[0],sub3x3TransposeCofactors[0]))&0x80000000u;
}

// use this if you anticipate flipped/mirrored models
vec3 fastNormalTransform(in uint signFlipMask, in mat3 sub3x3TransposeCofactors, in vec3 normal)
{
    vec3 tmp = sub3x3TransposeCofactors*normal;
    const float tmpLenRcp = inversesqrt(dot(tmp,tmp));
    return tmp*uintBitsToFloat(floatBitsToUint(tmpLenRcp)^signFlipMask);
}
#endif

//
template<typename Mat3x4> NBL_REQUIRES(is_matrix_v<Mat3x4>) // TODO: allow any matrix type AND our emulated ones
Mat3x4 pseudoInverse3x4(NBL_CONST_REF_ARG(Mat3x4) tform)
{
    const matrix<scalar_type_t<Mat3x4>,3,3> sub3x3Inv = inverse(mat3(tform));
    Mat3x4 retval;
    retval[0] = sub3x3Inv[0];
    retval[1] = sub3x3Inv[1];
    retval[2] = sub3x3Inv[2];
    retval[3] = -sub3x3Inv*tform[3];
    return retval;
}

#endif