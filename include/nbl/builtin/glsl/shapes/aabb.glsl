// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_SHAPES_AABB_INCLUDED_
#define _NBL_BUILTIN_GLSL_SHAPES_AABB_INCLUDED_

#ifndef __cplusplus
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


// returns the corner of the AABB which has the most positive dot product
vec3 nbl_glsl_shapes_AABB_getFarthestPointInFront(in nbl_glsl_shapes_AABB_t aabb, in vec3 plane)
{
    return mix(aabb.maxVx,aabb.minVx,lessThan(plane,vec3(0.f)));
}
#endif

struct nbl_glsl_shapes_CompressedAABB_t
{
    uvec2 minVx18E7S3;
    uvec2 maxVx18E7S3;
};
#ifndef __cplusplus
#include <nbl/builtin/glsl/format/decode.glsl>
nbl_glsl_shapes_AABB_t nbl_glsl_shapes_CompressedAABB_t_decompress(in nbl_glsl_shapes_CompressedAABB_t compressed)
{
    nbl_glsl_shapes_AABB_t retval;
    retval.minVx = nbl_glsl_decodeRGB18E7S3(compressed.minVx18E7S3);
    retval.maxVx = nbl_glsl_decodeRGB18E7S3(compressed.maxVx18E7S3);
    return retval;
}
#endif

#endif
