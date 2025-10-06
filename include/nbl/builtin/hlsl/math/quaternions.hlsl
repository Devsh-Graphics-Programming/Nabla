
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_MATH_QUATERNIONS_INCLUDED_
#define _NBL_BUILTIN_GLSL_MATH_QUATERNIONS_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace math
{


struct quaternion_t
{
    float4 data;


    static quaternion_t constructFromTruncated(in float3 first3Components)
    {
        quaternion_t quat;
        quat.data.xyz = first3Components;
        quat.data.w = sqrt(1.0-dot(first3Components,first3Components));
        return quat;
    }

    static quaternion_t lerp(in quaternion_t start, in quaternion_t end, in float fraction, in float totalPseudoAngle)
    {
        const uint negationMask = asuint(totalPseudoAngle) & 0x80000000u;
        const float4 adjEnd = asfloat(asuint(end.data)^negationMask);

        quaternion_t quat;
        quat.data = lerp(start.data, adjEnd, fraction);
        return quat;
    }
    static quaternion_t lerp(in quaternion_t start, in quaternion_t end, in float fraction)
    {
        return lerp(start,end,fraction,dot(start.data,end.data));
    }

    static float flerp_impl_adj_interpolant(in float angle, in float fraction, in float interpolantPrecalcTerm2, in float interpolantPrecalcTerm3)
    {
        const float A = 1.0904f + angle * (-3.2452f + angle * (3.55645f - angle * 1.43519f));
        const float B = 0.848013f + angle * (-1.06021f + angle * 0.215638f);
        const float k = A * interpolantPrecalcTerm2 + B;
        return fraction+interpolantPrecalcTerm3*k;
    }

    static quaternion_t flerp(in quaternion_t start, in quaternion_t end, in float fraction)
    {
        const float pseudoAngle = dot(start.data,end.data);

        const float interpolantPrecalcTerm = fraction-0.5f;
        const float interpolantPrecalcTerm3 = fraction*interpolantPrecalcTerm*(fraction-1.f);
        const float adjFrac = quaternion_t::flerp_impl_adj_interpolant(abs(pseudoAngle),fraction,interpolantPrecalcTerm*interpolantPrecalcTerm,interpolantPrecalcTerm3);
        quaternion_t quat = quaternion_t::lerp(start,end,adjFrac,pseudoAngle);
        quat.data = normalize(quat.data);
        return quat;
    }

    static float3x3 constructMatrix(in quaternion_t quat)
    {
        float3x3 mat;
        mat[0] = quat.data.yzx*quat.data.ywz+quat.data.zxy*quat.data.zyw*float3( 1.f, 1.f,-1.f);
        mat[1] = quat.data.yzx*quat.data.xzw+quat.data.zxy*quat.data.wxz*float3(-1.f, 1.f, 1.f);
        mat[2] = quat.data.yzx*quat.data.wyx+quat.data.zxy*quat.data.xwy*float3( 1.f,-1.f, 1.f);
        mat[0][0] = 0.5f-mat[0][0];
        mat[1][1] = 0.5f-mat[1][1];
        mat[2][2] = 0.5f-mat[2][2];
        mat *= 2.f;
        return mat;
    }
};

float3 slerp_delta_impl(in float3 start, in float3 preScaledWaypoint, in float cosAngleFromStart)
{
    float3 planeNormal = cross(start,preScaledWaypoint);
    
    cosAngleFromStart *= 0.5;
    const float sinAngle = sqrt(0.5-cosAngleFromStart);
    const float cosAngle = sqrt(0.5+cosAngleFromStart);
    
    planeNormal *= sinAngle;
    const float3 precompPart = cross(planeNormal,start)*2.0;

    return precompPart*cosAngle+cross(planeNormal,precompPart);
}

float3 slerp_impl_impl(in float3 start, in float3 preScaledWaypoint, in float cosAngleFromStart)
{
    return start + slerp_delta_impl(start,preScaledWaypoint,cosAngleFromStart);
}



}
}
}

#endif