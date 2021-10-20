// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_

bool nbl_glsl_fastCullAABBvsFrustum(in mat4 proj, in mat2x3 bbox)
{
    mat4 pTpose = transpose(proj);
    // xyPlanes[0] = NDC.x = -1 frustum plane oriented inwards
    // xyPlanes[1] = NDC.x = +1 frustum plane oriented inwards
    // xyPlanes[2] = NDC.y = -1 frustum plane oriented inwards
    // xyPlanes[3] = NDC.y = +1 frustum plane oriented inwards
    mat4 xyPlanes = mat4(pTpose[3]+pTpose[0],pTpose[3]-pTpose[0], pTpose[3]+pTpose[1],pTpose[3]-pTpose[1]);
    // far plane, oriented inwards (toward camera)
    vec4 farPlane = pTpose[3]-pTpose[2];

    // @Przemog, you can abuse this to find the screenspace NDC extents of an AABB under an MVP
#define getClosestDP(R) (dot(mix(bbox[1],bbox[0],lessThan(R.xyz,vec3(0.f)) ),R.xyz)+R.w)
    // you want to cull by screen side planes, because it throws away the most of the scene in all scene types
    // top down view = yes
    // open world = yes
    // random soup of objects/indoor = yes
    if (getClosestDP(xyPlanes[0])<=0.f || getClosestDP(xyPlanes[1])<=0.f)
        return true;
    
    // then unless you're doing a flight sim or a Strategy Game, you want to cull using the near plane
    // top down view = not much
    // open world = yes
    // random soup of objects/indoor = yes 
    if (getClosestDP(pTpose[3])<=0.f)
        return true;
 
    // the far plane never does much, so do the top/down planes, but do the bottom plane first
    // (a sane person doesn't create objects or load a scene past a certain distance) 
    // top down view = bottom gets you more
    // open world = bottom gets you more
    // random soup of objects/indoor = yes
    if (getClosestDP(xyPlanes[2])<=0.f || getClosestDP(xyPlanes[3])<=0.f)
        return true;

    return getClosestDP(farPlane)<=0.f;
#undef getClosestDP
}

#endif