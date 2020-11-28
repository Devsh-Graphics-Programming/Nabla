// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_

bool nbl_glsl_couldBeVisible(in mat4 proj, in mat2x3 bbox)
{
    mat4 pTpose = transpose(proj);
    mat4 xyPlanes = mat4(pTpose[3] + pTpose[0], pTpose[3] + pTpose[1], pTpose[3] - pTpose[0], pTpose[3] - pTpose[1]);
    vec4 farPlane = pTpose[3] + pTpose[2];

#define getClosestDP(R) (dot(mix(bbox[1],bbox[0],lessThan(R.xyz,vec3(0.f)) ),R.xyz)+R.w>0.f)

    return  getClosestDP(xyPlanes[0]) && getClosestDP(xyPlanes[1]) &&
        getClosestDP(xyPlanes[2]) && getClosestDP(xyPlanes[3]) &&
        getClosestDP(pTpose[3]) && getClosestDP(farPlane);
#undef getClosestDP
}

#endif