// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_

#include <nbl/builtin/glsl/shapes/aabb.glsl>
#include <nbl/builtin/glsl/shapes/frustum.glsl>

// gives false negatives
bool nbl_glsl_fastestFrustumCullAABB(in mat4 proj, in nbl_glsl_shapes_AABB_t aabb)
{
    const nbl_glsl_shapes_Frustum_t frust = nbl_glsl_shapes_Frustum_extract(proj);
    return nbl_glsl_shapes_Frustum_fastestDoesNotIntersectAABB(frust,aabb);
}

// gives very few false negatives
bool nbl_glsl_fastFrustumCullAABB(in mat4 proj, in mat4 invProj, in nbl_glsl_shapes_AABB_t aabb)
{
    if (nbl_glsl_fastestFrustumCullAABB(proj,aabb))
        return true;

    const nbl_glsl_shapes_Frustum_t boxInvFrustum = nbl_glsl_shapes_Frustum_extract(invProj);
    nbl_glsl_shapes_AABB_t ndc;
    ndc.minVx = vec3(-1.f,-1.f,0.f);
    ndc.maxVx = vec3(1.f,1.f,1.f);
    return nbl_glsl_shapes_Frustum_fastestDoesNotIntersectAABB(boxInvFrustum,ndc);
}

// perfect Separating Axis Theorem
/* Needed for Clustered/Tiled Lighting
bool nbl_glsl_preciseFrustumCullAABB(in mat4 proj, in mat4 invProj, in nbl_glsl_shapes_AABB_t aabb)
{
    const nbl_glsl_shapes_Frustum_t viewFrust = nbl_glsl_shapes_Frustum_extract(proj);
    if (nbl_glsl_shapes_Frustum_fastestDoesNotIntersectAABB(viewFrust,aabb))
        return true;
    
    const nbl_glsl_shapes_Frustum_t boxInvFrustum = nbl_glsl_shapes_Frustum_extract(invProj);
    nbl_glsl_shapes_AABB_t ndc;
    ndc.minVx = vec3(-1.f,-1.f,0.f);
    ndc.maxVx = vec3(1.f,1.f,1.f);
    if (nbl_glsl_shapes_Frustum_fastestDoesNotIntersectAABB(boxInvFrustum,ndc))
        return true;

    // TODO: extract all 8 vertices of the frustum given invProj
    vec3 vertices[8];
    vec3 edges[12];
    edges[ 0] = cross(viewFrust.minPlanes[0].xyz,viewFrust.minPlanes[1].xyz);
    edges[ 1] = cross(viewFrust.minPlanes[0].xyz,viewFrust.minPlanes[2].xyz);
    edges[ 2] = cross(viewFrust.minPlanes[0].xyz,viewFrust.maxPlanes[1].xyz);
    edges[ 3] = cross(viewFrust.minPlanes[0].xyz,viewFrust.maxPlanes[2].xyz);
    edges[ 4] = cross(viewFrust.minPlanes[1].xyz,viewFrust.minPlanes[0].xyz);
    edges[ 5] = cross(viewFrust.minPlanes[1].xyz,viewFrust.minPlanes[2].xyz);
    edges[ 6] = cross(viewFrust.minPlanes[1].xyz,viewFrust.maxPlanes[0].xyz);
    edges[ 7] = cross(viewFrust.minPlanes[1].xyz,viewFrust.maxPlanes[2].xyz);
    edges[ 8] = cross(viewFrust.minPlanes[2].xyz,viewFrust.minPlanes[0].xyz);
    edges[ 0] = cross(viewFrust.minPlanes[2].xyz,viewFrust.minPlanes[1].xyz);
    edges[10] = cross(viewFrust.minPlanes[2].xyz,viewFrust.maxPlanes[0].xyz);
    edges[11] = cross(viewFrust.minPlanes[2].xyz,viewFrust.maxPlanes[1].xyz);
    for (int i=0; i<12; i++)
    {
        // cross(e_0,edges[i])
        {
            const vec2 normal = vec2(-edges[i].z,edges[i].y);
            const bvec2 negMask = lessThan(normal,vec2(0.f));
#define getDP(V) dot(V.yz,normal)
            const float minAABB = dot(mix(aabb.minVx.yz,aabb.maxVx.yz,negMask),normal);
            for (int j=1; j<8; j++)
            if (getDP(vertices[j])<=minAABB)
                return true;
            const float maxAABB = dot(mix(aabb.maxVx.yz,aabb.minVx.yz,negMask),normal);
            for (int j=1; j<8; j++)
            if (maxAABB<=getDP(vertices[j]))
                return true;
#undef getDP
        }
        // cross(e_1,edges[i])
        {
            const vec2 normal = vec2(-edges[i].x,edges[i].z);
            const bvec2 negMask = lessThan(normal,vec2(0.f));
#define getDP(V) dot(V.xz,normal)
            const float minAABB = dot(mix(aabb.minVx.xz,aabb.maxVx.xz,negMask),normal);
            for (int j=1; j<8; j++)
            if (getDP(vertices[j])<=minAABB)
                return true;
            const float maxAABB = dot(mix(aabb.maxVx.xz,aabb.minVx.xz,negMask),normal);
            for (int j=1; j<8; j++)
            if (maxAABB<=getDP(vertices[j]))
                return true;
#undef getDP
        }
        // cross(e_2,edges[i])
        {
            const vec2 normal = vec2(-edges[i].y,edges[i].x);
            const bvec2 negMask = lessThan(normal,vec2(0.f));
#define getDP(V) dot(V.xy,normal)
            const float minAABB = dot(mix(aabb.minVx.xy,aabb.maxVx.xy,negMask),normal);
            for (int j=1; j<8; j++)
            if (getDP(vertices[j])<=minAABB)
                return true;
            const float maxAABB = dot(mix(aabb.maxVx.xy,aabb.minVx.xy,negMask),normal);
            for (int j=1; j<8; j++)
            if (maxAABB<=getDP(vertices[j]))
                return true;
#undef getDP
        }
    }
    return false;
}
*/

// TODO: Other culls useful for clustered lighting
// - Sphere vs Frustum
// - Convex Infinite Cone vs Frustum
// - Concave Infinite Cone vs Frustum (! is frustum inside of an convex infinite cone with half angle PI-theta)


#endif