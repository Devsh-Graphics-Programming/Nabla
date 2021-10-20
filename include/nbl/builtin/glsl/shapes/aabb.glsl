// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_SHAPES_AABB_INCLUDED_
#define _NBL_BUILTIN_GLSL_SHAPES_AABB_INCLUDED_

struct nbl_glsl_shapes_AABB_t
{
    vec3 minVx;
    vec3 maxVx;
};

void nbl_glsl_shapes_AABB_addPoint(in nbl_glsl_shapes_AABB_t aabb, in vec3 pt)
{
    aabb.minVx = min(pt,aabb.minVx);
    aabb.maxVx = max(pt,aabb.maxVx);
}

vec3 nbl_glsl_shapes_AABB_getExtent(in nbl_glsl_shapes_AABB_t aabb)
{
    return aabb.maxVx-aabb.minVx;
}
float nbl_glsl_shapes_AABB_getVolume(in nbl_glsl_shapes_AABB_t aabb)
{
    const vec3 extent = nbl_glsl_shapes_AABB_getExtent(aabb);
    return extent.x*extent.y*extent.z;
}

#endif
