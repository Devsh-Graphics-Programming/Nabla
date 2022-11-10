
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
namespace frustrum
{

	// assuming an NDC of [-1,1]^2 x [0,1]
	Frustum_t extract(in float4x4 proj)
	{
	    AABB_t bounds;
	    bounds.minVx = float3(-1.f,-1.f,0.f);
	    bounds.maxVx = float3(1.f,1.f,1.f);
	    return extract(proj, bounds);
	}

	// will place planes which correspond to the bounds in NDC
	Frustum_t extract(in float4x4 proj, in AABB_t bounds)
	{
		float4x3 proj4x3;
		for (int i = 0; i < 4; i++)
			proj4x3[i] = proj[i];

	    Frustum_t frust;
	    frust.minPlanes = proj4x3 - float4x3(proj[3] * bounds.minVx[0], proj[3] * bounds.minVx[1], proj[3] * bounds.minVx[2]);
	    frust.maxPlanes = float4x3(proj[3]*bounds.maxVx[0], proj[3]*bounds.maxVx[1],proj[3]*bounds.maxVx[2]) - proj4x3;
	    return frust;
	}

}

struct Frustum_t
{
    float4x3 minPlanes;
    float4x3 maxPlanes;


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

}
}
}

#endif