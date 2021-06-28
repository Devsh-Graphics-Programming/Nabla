// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_
#define _NBL_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_


vec3 nbl_glsl_perturbNormal_heightMap(in vec3 vtxN, in mat2x3 dPdQ, in vec2 dHdQ)
{
    vec3 r1 = cross(dPdQ[1],vtxN);
    vec3 r2 = cross(vtxN,dPdQ[0]);
    vec3 surfGrad = (r1*dHdQ.x+r2*dHdQ.y)/dot(dPdQ[0],r1);
    return normalize(vtxN-surfGrad);
}

vec3 nbl_glsl_perturbNormal_derivativeMap(in vec3 vtxN, in vec2 dhdUV, in mat2x3 dPdQ, in mat2 dUVdQ)
{
    // apply the chain rule
    const vec2 dHdQ = vec2(dot(dhdUV,dUVdQ[0]), dot(dhdUV,dUVdQ[1]));
    return nbl_glsl_perturbNormal_heightMap(vtxN,dPdQ,dHdQ);
}

#ifdef _NBL_BUILTIN_GLSL_BUMP_MAPPING_DERIVATIVES_DECLARED_
vec3 nbl_glsl_perturbNormal_derivativeMap(in vec3 vtxN, in vec2 dhdUV)
{
    return nbl_glsl_perturbNormal_derivativeMap(vtxN,dhdUV,nbl_glsl_perturbNormal_dPdSomething(),nbl_glsl_perturbNormal_dUVdSomething());
}
#endif

#endif