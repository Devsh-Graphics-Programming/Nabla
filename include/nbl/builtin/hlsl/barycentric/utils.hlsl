// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BARYCENTRIC_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BARYCENTRIC_UTILS_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace barycentric
{

float2 reconstructBarycentrics(in float3 positionRelativeToV0, in float2x3 edges)
{
    const float e0_2 = dot(edges[0], edges[0]);
    const float e0e1 = dot(edges[0], edges[1]);
    const float e1_2 = dot(edges[1], edges[1]);

    const float qe0 = dot(positionRelativeToV0, edges[0]);
    const float qe1 = dot(positionRelativeToV0, edges[1]);
    const float2 protoBary = float2(qe0 * e1_2 - qe1 * e0e1, qe1 * e0_2 - qe0 * e0e1);

    const float rcp_dep = 1.f / (e0_2 * e1_2 - e0e1 * e0e1);
    return protoBary * rcp_dep;
}
float2 reconstructBarycentrics(in float3 pointPosition, in float3x3 vertexPositions)
{
    return reconstructBarycentrics(pointPosition - vertexPositions[2], float2x3(vertexPositions[0] - vertexPositions[2], vertexPositions[1] - vertexPositions[2]));
}

float3 expand(in float2 compactBarycentrics)
{
    return float3(compactBarycentrics.xy,1.f-compactBarycentrics.x-compactBarycentrics.y);
}

}
}
}

#endif
