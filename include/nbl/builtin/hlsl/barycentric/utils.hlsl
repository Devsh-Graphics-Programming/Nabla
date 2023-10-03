// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BARYCENTRIC_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BARYCENTRIC_UTILS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{
namespace barycentric
{

float32_t2 reconstructBarycentrics(NBL_CONST_REF_ARG(float32_t3) positionRelativeToV0, NBL_CONST_REF_ARG(float32_t2x3) edges)
{
    const float32_t e0_2 = dot(edges[0], edges[0]);
    const float32_t e0e1 = dot(edges[0], edges[1]);
    const float32_t e1_2 = dot(edges[1], edges[1]);

    const float32_t qe0 = dot(positionRelativeToV0, edges[0]);
    const float32_t qe1 = dot(positionRelativeToV0, edges[1]);
    const float32_t2 protoBary = float32_t2(qe0 * e1_2 - qe1 * e0e1, qe1 * e0_2 - qe0 * e0e1);

    const float32_t rcp_dep = 1.f / (e0_2 * e1_2 - e0e1 * e0e1);
    return protoBary * rcp_dep;
}
float32_t2 reconstructBarycentrics(NBL_CONST_REF_ARG(float32_t3) pointPosition, NBL_CONST_REF_ARG(float32_t3x3) vertexPositions)
{
    return reconstructBarycentrics(pointPosition - vertexPositions[2], float32_t2x3(vertexPositions[0] - vertexPositions[2], vertexPositions[1] - vertexPositions[2]));
}

float32_t3 expand(NBL_CONST_REF_ARG(float32_t2) compactBarycentrics)
{
    return float32_t3(compactBarycentrics.xy,1.f-compactBarycentrics.x-compactBarycentrics.y);
}

}
}
}

#endif
