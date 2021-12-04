// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_
#define _NBL_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_


vec3 nbl_glsl_perturbNormal_heightMap(in vec3 vtxN, in mat2x3 dPdQ, in vec2 dHdQ)
{
    const vec3 r1 = normalize(cross(vtxN,dPdQ[1]));
    const vec3 r2 = normalize(cross(dPdQ[0],vtxN));
    const float cosInPlane = dot(r1,r2);
    const vec3 surfGrad = (r1*dHdQ.x+r2*dHdQ.y)/inversesqrt(1.f-cosInPlane*cosInPlane);
    return normalize(vtxN-surfGrad);
}

vec3 nbl_glsl_perturbNormal_derivativeMap(in vec3 vtxN, in vec2 dhdUV, in mat2x3 dPdQ, in mat2 dUVdQ)
{
    // apply the chain rule
    const vec2 dUVdQ1 = normalize(dUVdQ[0]);
    const vec2 dUVdQ2 = normalize(dUVdQ[1]);
    const float cs = dot(dUVdQ1,dUVdQ2);
    const vec2 dHdQ = vec2(dot(dhdUV,dUVdQ1),dot(dhdUV,dUVdQ2))/inversesqrt(1.0-cs*cs);
    return nbl_glsl_perturbNormal_heightMap(vtxN,dPdQ,dHdQ);
}

#ifdef _NBL_BUILTIN_GLSL_BUMP_MAPPING_DERIVATIVES_DECLARED_
vec3 nbl_glsl_perturbNormal_derivativeMap(in vec3 vtxN, in vec2 dhdUV)
{
    /*
    mat2x3 dPdQ = nbl_glsl_perturbNormal_dPdSomething();
    mat2 dUVdQ = nbl_glsl_perturbNormal_dUVdSomething();

	mat2x3 dPdUV = dPdQ*inverse(dUVdQ);
	
    const vec3 r1 = cross(vtxN,dPdUV[1]);
    const float r1len2 = dot(r1,r1);
    const vec3 r2 = cross(dPdUV[0],vtxN);
    const float r2len2 = dot(r2,r2);

    vec3 surfGrad = (r1*sqrt(r2len2)*dhdUV.x + r2*sqrt(r1len2)*dhdUV.y)/inversesqrt(r2len2*r1len2-dot(r1,r2)*dot(r1,r2));
    return normalize(vtxN + surfGrad);
    */
    return nbl_glsl_perturbNormal_derivativeMap(vtxN,dhdUV,nbl_glsl_perturbNormal_dPdSomething(),nbl_glsl_perturbNormal_dUVdSomething());
}
#endif

#endif