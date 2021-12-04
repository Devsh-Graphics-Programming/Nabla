// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_
#define _NBL_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_


vec3 nbl_glsl_perturbNormal_heightMap(in vec3 vtxN, in vec2 dhdUV, in mat2x3 dPdUV)
{
    // no idea if this is correct, but seems to work
    const vec3 r1 = normalize(cross(vtxN,dPdUV[1]));
    const vec3 r2 = normalize(cross(dPdUV[0],vtxN));
    const float cosInPlane = dot(r1,r2);
    const vec3 surfGrad = r1*dhdUV.x+r2*dhdUV.y;
    return normalize(vtxN*sqrt(1.f-cosInPlane*cosInPlane)+surfGrad);
}

vec3 nbl_glsl_perturbNormal_derivativeMap(in vec3 vtxN, in vec2 dhdUV, in mat2x3 dPdQ, in mat2 dUVdQ)
{
    // apply the chain rule in reverse
    const mat2x3 dPdUV = dPdQ*inverse(dUVdQ);
    return nbl_glsl_perturbNormal_heightMap(vtxN,dhdUV,dPdUV);
}

#ifdef _NBL_BUILTIN_GLSL_BUMP_MAPPING_DERIVATIVES_DECLARED_
vec3 nbl_glsl_perturbNormal_derivativeMap(in vec3 vtxN, in vec2 dhdUV)
{
    return nbl_glsl_perturbNormal_derivativeMap(vtxN,dhdUV,nbl_glsl_perturbNormal_dPdSomething(),nbl_glsl_perturbNormal_dUVdSomething());
}
#endif

#endif