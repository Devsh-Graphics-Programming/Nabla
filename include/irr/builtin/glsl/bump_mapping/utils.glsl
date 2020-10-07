// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _IRR_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_
#define _IRR_BUILTIN_GLSL_BUMP_MAPPING_UTILS_INCLUDED_

vec3 irr_glsl_perturbNormal_heightMap(in vec3 vtxN, in mat2x3 dPdScreen, in vec2 dHdScreen)
{
    vec3 r1 = cross(dPdScreen[1], vtxN);
    vec3 r2 = cross(vtxN, dPdScreen[0]);
    vec3 surfGrad = (r1 * dHdScreen.x + r2 * dHdScreen.y) / dot(dPdScreen[0], r1);
    return normalize(vtxN - surfGrad);
}

vec3 irr_glsl_perturbNormal_derivativeMap(in vec3 normal, in vec2 dh, in mat2x3 dPdScreen, in mat2 dUVdScreen)
{
    vec2 dHdScreen = vec2(dot(dh, dUVdScreen[0]), dot(dh, dUVdScreen[1]));//apply chain rule

    return irr_glsl_perturbNormal_heightMap(normal, dPdScreen, dHdScreen);
}

#endif