// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_SHAPES_FRUSTUM_INCLUDED_
#define _NBL_BUILTIN_GLSL_SHAPES_FRUSTUM_INCLUDED_

#include <nbl/builtin/glsl/shapes/aabb.glsl>

struct nbl_glsl_shapes_Frustum_t
{
    mat3x4 minPlanes;
    mat3x4 maxPlanes;
};

// will place planes which correspond to the bounds in NDC
nbl_glsl_shapes_Frustum_t nbl_glsl_shapes_Frustum_extract(in mat4 proj, in nbl_glsl_shapes_AABB_t bounds)
{
    const mat4 pTpose = transpose(proj);

    nbl_glsl_shapes_Frustum_t frust;
    frust.minPlanes = mat3x4(pTpose)-mat3x4(pTpose[3]*bounds.minVx[0],pTpose[3]*bounds.minVx[1],pTpose[3]*bounds.minVx[2]);
    frust.maxPlanes = mat3x4(pTpose[3]*bounds.maxVx[0],pTpose[3]*bounds.maxVx[1],pTpose[3]*bounds.maxVx[2])-mat3x4(pTpose);
    return frust;
}

// assuming an NDC of [-1,1]^2 x [0,1]
nbl_glsl_shapes_Frustum_t nbl_glsl_shapes_Frustum_extract(in mat4 proj)
{
    nbl_glsl_shapes_AABB_t bounds;
    bounds.minVx = vec3(-1.f,-1.f,0.f);
    bounds.maxVx = vec3(1.f,1.f,1.f);
    return nbl_glsl_shapes_Frustum_extract(proj,bounds);
}

// gives false negatives
bool nbl_glsl_shapes_Frustum_fastestDoesNotIntersectAABB(in nbl_glsl_shapes_Frustum_t frust, in nbl_glsl_shapes_AABB_t aabb)
{
#define getClosestDP(R) (dot(nbl_glsl_shapes_AABB_getFarthestPointInFront(aabb,R.xyz),R.xyz)+R.w)
    if (getClosestDP(frust.minPlanes[0])<=0.f)
        return true;
    if (getClosestDP(frust.minPlanes[1])<=0.f)
        return true;
    if (getClosestDP(frust.minPlanes[2])<=0.f)
        return true;

    if (getClosestDP(frust.maxPlanes[0])<=0.f)
        return true;
    if (getClosestDP(frust.maxPlanes[1])<=0.f)
        return true;
    
    return getClosestDP(frust.maxPlanes[2])<=0.f;
#undef getClosestDP
}

#endif
