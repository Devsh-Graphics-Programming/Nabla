// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_
#define _NBL_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_


vec3 nbl_glsl_perturbNormal_heightMap(in vec3 vtxN, in vec2 dhdUV, in mat2x3 dPdUV)
{
    // no idea if this is correct, but seems to work
    const vec3 r1 = cross(vtxN,dPdUV[1]);
    const vec3 r2 = cross(dPdUV[0],vtxN);
    const float r1len2 = dot(r1,r1);
    const float r2len2 = dot(r2,r2);
    const float cosInPlane = dot(r1,r2);
    // protect against zero length r1 or r2, colinear r1 and r2, and NaN
    const float sinInPlane2 = r1len2*r2len2-cosInPlane*cosInPlane;
    if (sinInPlane2>0.0000001f)
    { 
        const vec3 surfGrad = r1*sqrt(r2len2)*dhdUV.x+r2*sqrt(r1len2)*dhdUV.y;
        return normalize(vtxN*sqrt(sinInPlane2)+surfGrad);
    }
    return vtxN;
}

vec3 nbl_glsl_perturbNormal_derivativeMap(in vec3 vtxN, in vec2 dhdUV, in mat2x3 dPdQ, mat2 dUVdQ)
{
    // apply the chain rule in reverse
	dUVdQ /= abs(determinant(dUVdQ));
    mat2x3 dPdUV;
	dPdUV[0] = dPdQ[0]*dUVdQ[1][1]-dPdQ[1]*dUVdQ[0][1];
	dPdUV[1] = dPdQ[1]*dUVdQ[0][0]-dPdQ[0]*dUVdQ[1][0];
    return nbl_glsl_perturbNormal_heightMap(vtxN,dhdUV,dPdUV);
}

#ifdef _NBL_BUILTIN_GLSL_BUMP_MAPPING_DERIVATIVES_DECLARED_
vec3 nbl_glsl_perturbNormal_derivativeMap(in vec3 vtxN, in vec2 dhdUV)
{
    return nbl_glsl_perturbNormal_derivativeMap(vtxN,dhdUV,nbl_glsl_perturbNormal_dPdSomething(),nbl_glsl_perturbNormal_dUVdSomething());
}
#endif

#endif