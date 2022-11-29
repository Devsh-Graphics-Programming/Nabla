
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SHAPES_FRUSTUM_INCLUDED_
#define _NBL_BUILTIN_HLSL_SHAPES_FRUSTUM_INCLUDED_

#include <nbl/builtin/hlsl/shapes/aabb.hlsl>


namespace nbl
{
namespace hlsl
{
namespace shapes
{


struct Frustum_t
{
    float3x4 minPlanes;
    float3x4 maxPlanes;


	// gives false negatives
	bool fastestDoesNotIntersectAABB(in Frustum_t frust, in AABB_t aabb)
	{
	#define getClosestDP(R) (dot(aabb.getFarthestPointInFront(R.xyz), R.xyz) + R.w)
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
};


// will place planes which correspond to the bounds in NDC
Frustum_t extract(in float4x4 proj, in AABB_t bounds)
{
    const float4x4 pTpose = transpose(proj);

    Frustum_t frust;
    frust.minPlanes = (float3x4)(pTpose) - float3x4(pTpose[3]*bounds.minVx[0], pTpose[3]*bounds.minVx[1], pTpose[3]*bounds.minVx[2]);
    frust.maxPlanes = float3x4(pTpose[3]*bounds.maxVx[0], pTpose[3]*bounds.maxVx[1], pTpose[3]*bounds.maxVx[2]) - (float3x4)(pTpose);
    return frust;
}

// assuming an NDC of [-1,1]^2 x [0,1]
Frustum_t extract(in float4x4 proj)
{
    AABB_t bounds;
    bounds.minVx = float3(-1.f,-1.f,0.f);
    bounds.maxVx = float3(1.f,1.f,1.f);
    return extract(proj, bounds);
}


}
}
}

#endif