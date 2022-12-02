
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_
#define _NBL_BUILTIN_GLSL_UTILS_CULLING_INCLUDED_

#include <nbl/builtin/hlsl/shapes/aabb.hlsl>
#include <nbl/builtin/hlsl/shapes/frustum.hlsl>


namespace nbl
{
namespace hlsl
{


// gives false negatives
bool fastestFrustumCullAABB(in float4x4 proj, in shapes::AABB_t aabb)
{
    const shapes::Frustum_t frust = shapes::Frustum_t::extract(proj);
    return shapes::Frustum_t::fastestDoesNotIntersectAABB(frust, aabb);
}

// gives very few false negatives
bool fastFrustumCullAABB(in float4x4 proj, in float4x4 invProj, in shapes::AABB_t aabb)
{
    if (fastestFrustumCullAABB(proj,aabb))
        return true;

    const shapes::Frustum_t boxInvFrustum = shapes::Frustum_t::extract(invProj);
    shapes::AABB_t ndc;
    ndc.minVx = float3(-1.f,-1.f,0.f);
    ndc.maxVx = float3(1.f,1.f,1.f);
    return shapes::Frustum_t::fastestDoesNotIntersectAABB(boxInvFrustum,ndc);
}

// perfect Separating Axis Theorem, needed for Clustered/Tiled Lighting
bool preciseFrustumCullAABB(in float4x4 proj, in float4x4 invProj, in shapes::AABB_t aabb)
{
    const shapes::Frustum_t viewFrust = shapes::Frustum_t::extract(proj);
    if (shapes::Frustum_t::fastestDoesNotIntersectAABB(viewFrust,aabb))
        return true;
    
    const shapes::Frustum_t boxInvFrustum = shapes::Frustum_t::extract(invProj);
    shapes::AABB_t ndc;
    ndc.minVx = float3(-1.f,-1.f,0.f);
    ndc.maxVx = float3(1.f,1.f,1.f);
    if (shapes::Frustum_t::fastestDoesNotIntersectAABB(boxInvFrustum,ndc))
        return true;

    float3 edges[12];
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
#define getClosestDP(R) (dot(ndc.getFarthestPointInFront(R.xyz),R.xyz)+R.w)
        /* TODO: These are buggy!
        // cross(e_0,edges[i])
        {
            const float2 normal = float2(-edges[i].z,edges[i].y);
            const bool2 negMask = lessThan(normal,float2(0.f));
            const float4 planeBase = normal.x*invProj[1]+normal.y*invProj[2];
            
            const float minAABB = dot(lerp(aabb.minVx.yz,aabb.maxVx.yz,negMask),normal);
            const float4 minPlane = planeBase-invProj[3]*minAABB;
            if (getClosestDP(minPlane)<=0.f)
                return true;
            const float maxAABB = dot(lerp(aabb.maxVx.yz,aabb.minVx.yz,negMask),normal);
            const float4 maxPlane = invProj[3]*maxAABB-planeBase;
            if (getClosestDP(maxPlane)<=0.f)
                return true;
        }
        // cross(e_1,edges[i])
        {
            const float2 normal = float2(-edges[i].x,edges[i].z);
            const bool2 negMask = lessThan(normal,float2(0.f));
            const float4 planeBase = normal.x*invProj[0]+normal.y*invProj[2];
            const float minAABB = dot(lerp(aabb.minVx.xz,aabb.maxVx.xz,negMask),normal);
            const float4 minPlane = planeBase-invProj[3]*minAABB;
            if (getClosestDP(minPlane)<=0.f)
                return true;
            const float maxAABB = dot(lerp(aabb.maxVx.xz,aabb.minVx.xz,negMask),normal);
            const float4 maxPlane = invProj[3]*maxAABB-planeBase;
            if (getClosestDP(maxPlane)<=0.f)
                return true;
        } the last one is probably buggy too*/
        // cross(e_2,edges[i])
        {
            const float2 normal = float2(-edges[i].y,edges[i].x);
            const bool2 negMask = normal < (0.0f).xx;
            const float4 planeBase = normal.x*invProj[0]+normal.y*invProj[1];

            const float minAABB = dot(lerp(aabb.minVx.xy,aabb.maxVx.xy,negMask),normal);
            const float4 minPlane = planeBase-invProj[3]*minAABB;
            if (getClosestDP(minPlane)<=0.f)
                return true;
            const float maxAABB = dot(lerp(aabb.maxVx.xy,aabb.minVx.xy,negMask),normal);
            const float4 maxPlane = invProj[3]*maxAABB-planeBase;
            if (getClosestDP(maxPlane)<=0.f)
                return true;
        }
#undef getClosestDP
    }
    return false;
}

// TODO: Other culls useful for clustered lighting
// - Sphere vs Frustum
// - Convex Infinite Cone vs Frustum
// - Concave Infinite Cone vs Frustum (! is frustum inside of an convex infinite cone with half angle PI-theta)


}
}


#endif