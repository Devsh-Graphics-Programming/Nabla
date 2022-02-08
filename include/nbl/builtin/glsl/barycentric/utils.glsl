// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BARYCENTRIC_UTILS_INCLUDED_
#define _NBL_BUILTIN_GLSL_BARYCENTRIC_UTILS_INCLUDED_

vec2 nbl_glsl_barycentric_reconstructBarycentrics(in vec3 positionRelativeToV0, in mat2x3 edges)
{
    const float e0_2 = dot(edges[0],edges[0]);
    const float e0e1 = dot(edges[0],edges[1]);
    const float e1_2 = dot(edges[1],edges[1]);

    const float qe0 = dot(positionRelativeToV0,edges[0]);
    const float qe1 = dot(positionRelativeToV0,edges[1]);
    const vec2 protoBary = vec2(qe0*e1_2-qe1*e0e1,qe1*e0_2-qe0*e0e1);

    const float rcp_dep = 1.f/(e0_2*e1_2-e0e1*e0e1);
    return protoBary*rcp_dep;
}
vec2 nbl_glsl_barycentric_reconstructBarycentrics(in vec3 pointPosition, in mat3 vertexPositions)
{
    return nbl_glsl_barycentric_reconstructBarycentrics(pointPosition-vertexPositions[2],mat2x3(vertexPositions[0]-vertexPositions[2],vertexPositions[1]-vertexPositions[2]));
}

vec3 nbl_glsl_barycentric_expand(in vec2 compactBarycentrics)
{
    return vec3(compactBarycentrics.xy,1.0-compactBarycentrics.x-compactBarycentrics.y);
}

#endif