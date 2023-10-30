
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_UTILS_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_UTILS_TRANSFORM_INCLUDED_

#include <nbl/builtin/hlsl/limits/numeric.hlsl>


namespace nbl
{
namespace hlsl
{
namespace transform
{


// move to ieee754 header?
float3x3 mul_with_bounds_wo_gamma(out float3x3 error, in float3x3 a, in float3x3 b, in float b_relative_error)
{
    float3x3 retval;
    for (int i=0; i<3; i++)
    {
        float3 tmp = a[0]*b[i][0];
        retval[i] = tmp;
        error[i] = abs(tmp);
        for (int j=1; j<3; j++)
        {
            tmp = a[j]*b[i][j];
            retval[i] += tmp;
            error[i] += abs(tmp);
        }
    }
    const float error_factor = 1.f+b_relative_error/numeric_limits::float_epsilon(2);
    error *= error_factor;
    return retval;
}
float3x3 mul_with_bounds(out float3x3 error, in float3x3 a, in float3x3 b, in float b_relative_error)
{
    float3x3 retval = mul_with_bounds_wo_gamma(error,a,b,b_relative_error);
    error *= ieee754::gamma(2u);
    return retval;
}


/*
float4 pseudoMul4x4with3x1(in float4x4 m, in float3 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
}
float3 pseudoMul3x4with3x1(in float4x3 m, in float3 v)
{
    return m[0] * v.x + m[1] * v.y + m[2] * v.z + m[3];
}
float4x3 pseudoMul4x3with4x3(in float4x3 lhs, in float4x3 rhs) // TODO: change name to 3x4with3x4
{
    float4x3 result;
    for (int i = 0; i < 4; i++)
        result[i] = lhs[0] * rhs[i][0] + lhs[1] * rhs[i][1] + lhs[2] * rhs[i][2];
    result[3] += lhs[3];
    return result;
}
float4 pseudoMul4x4with4x3(in float4 proj, in float4x3 tform)
{
    float4 result;
    for (int i = 0; i < 4; i++)
        result[i] = proj[0] * tform[i][0] + proj[1] * tform[i][1] + proj[2] * tform[i][2];
    result[3] += proj[3];
    return result;
}
*/


// useful for fast computation of a Normal Matrix (you just need to remember to normalize the transformed normal because of the missing divide by the determinant)
float3x3 sub3x3TransposeCofactors_fn(in float3x3 sub3x3)
{
    return float3x3(
        cross(sub3x3[1],sub3x3[2]),
        cross(sub3x3[2],sub3x3[0]),
        cross(sub3x3[0],sub3x3[1])
    );
}
// returns a signflip mask
uint sub3x3TransposeCofactors_fn(in float3x3 sub3x3, out float3x3 sub3x3TransposeCofactors)
{
    sub3x3TransposeCofactors = sub3x3TransposeCofactors_fn(sub3x3);
    return asuint(dot(sub3x3[0], sub3x3TransposeCofactors[0])) & 0x80000000u;
}

// use this if you anticipate flipped/mirrored models
float3 fastNormalTransform(in uint signFlipMask, in float3x3 sub3x3TransposeCofactors, in float3 normal)
{
    float3 tmp = mul(sub3x3TransposeCofactors, normal);
    const float tmpLenRcp = rsqrt(dot(tmp,tmp));
    return tmp * asfloat(asuint(tmpLenRcp)^signFlipMask);
}

//
float4x3 pseudoInverse3x4(in float4x3 tform)
{
    const float3x3 sub3x3Inv = transpose(float3x3(tform[0], tform[1], tform[2]));
    float4x3 retval;
    retval[0] = sub3x3Inv[0];
    retval[1] = sub3x3Inv[1];
    retval[2] = sub3x3Inv[2];
    retval[3] = mul(-sub3x3Inv, tform[3]);
    return retval;
}


}
}
}

#endif