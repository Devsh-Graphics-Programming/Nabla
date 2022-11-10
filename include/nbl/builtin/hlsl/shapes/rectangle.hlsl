
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/math/functions.hlsl>


namespace nbl
{
namespace hlsl
{
namespace shapes
{


float3 getSphericalRectangle(in float3 observer, in float3 rectangleOrigin, in float3x3 rectangleNormalBasis)
{
    return mul((rectangleOrigin-observer), rectangleNormalBasis);
}

float SolidAngleOfRectangle(in float3 r0, in float2 rectangleExtents) 
{
    const float4 denorm_n_z = float4(-r0.y, r0.x+rectangleExtents.x, r0.y+rectangleExtents.y, -r0.x);
    const float4 n_z = denorm_n_z*rsqrt(float4(r0.z*r0.z,r0.z*r0.z,r0.z*r0.z,r0.z*r0.z)+denorm_n_z*denorm_n_z);
    const float4 cosGamma = float4(
        -n_z[0]*n_z[1],
        -n_z[1]*n_z[2],
        -n_z[2]*n_z[3],
        -n_z[3]*n_z[0]
    );
    return getSumofArccosABCD(cosGamma[0], cosGamma[1], cosGamma[2], cosGamma[3]) - 2*PI;
}


}
}
}

#endif