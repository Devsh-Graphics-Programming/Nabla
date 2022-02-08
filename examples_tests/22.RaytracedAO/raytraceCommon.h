#ifndef _RAYTRACE_COMMON_H_INCLUDED_
#define _RAYTRACE_COMMON_H_INCLUDED_

#include "common.h"

#if WORKGROUP_SIZE!=256
	#error "Hardcoded 16 should be NBL_SQRT(WORKGROUP_SIZE)"
#endif
#define WORKGROUP_DIM 16

/**
Plan for lighting:

Path Guiding with Rejection Sampling
	Do path guiding with spatio-directional (directions are implicit from light IDs) acceleration structure, could be with complete disregard for light NEE.
	Obviously the budgets for directions are low, so we might need to only track important lights and group them. Should probably read the spatiotemporal reservoir sampling paper.

	Each light gets a computed OBB and we use spherical OBB sampling (not projected solid angle, but we could clip) to generate the samples.
	Then NEE does perfect spherical sampling of the bounding volume.
	
	The OBBs could be hierarchical, possibly.

	OPTIMIZATION: Could possibly shoot an AnyHit to the front of the convex hull volume, and then ClosestHit between the front and back.
	BRDF sampling just samples the BSDF analytically (or gives up and samples only the path-guiding AS), uses Closest Hit and proceeds classically.
	There's essentially 3 ways to generate samples: NEE with PGAS (discrete directions), NEE with PGAS (for all incoming lights), BSDF Analytical.
	PROS: Probably a much better sample generation strategy, might clean up a lot of noise.
	CONS: We don't know the point on the surface we are going to hit (could be any of multiple points for a concave light), so we cannot cast a fixed length ray.
	We need to cast a ray to the furthest back side of the Bounding Volume, and it cannot be an just an AnyHit ray, it needs to have a ClosestHit shader that will compare
	if the hit instanceID==lightGroupID. It can probably be optimized so that it uses a different shadow-only + light-compare SBT. So it may take a lot longer to compute a sample.
CONCLUSION:
	We'll either be generating samples:
		A) From PGAS CDF
			No special light structure, just PGAS + GAS.
		C) Spherical sampling of OBBs
			OBB List with a CDF for the whole list in PGAS, then analytical

	Do we have to do 3-way MIS?
**/


struct SLight
{
#ifdef __cplusplus
	SLight() : obb() {}
	SLight(const SLight& other) : obb(other.obb) {}
	SLight(const nbl::core::aabbox3df& bbox, const nbl::core::matrix3x4SIMD& tform) : SLight()
	{
		auto extent = bbox.getExtent();
		obb.setScale(nbl::core::vectorSIMDf(extent.X, extent.Y, extent.Z));
		obb.setTranslation(nbl::core::vectorSIMDf(bbox.MinEdge.X, bbox.MinEdge.Y, bbox.MinEdge.Z));

		obb = nbl::core::concatenateBFollowedByA(tform, obb);
	}

	inline SLight& operator=(SLight&& other) noexcept
	{
		std::swap(obb, other.obb);

		return *this;
	}

	// also known as an upper bound on lumens put into the scene
	inline float computeFluxBound(const nbl::core::vectorSIMDf& radiance) const
	{
		const nbl::core::vectorSIMDf rec709LumaCoeffs(0.2126f, 0.7152f, 0.0722f, 0.f);
		const auto unitHemisphereArea = 2.f * nbl::core::PI<float>();

		const auto unitBoxScale = obb.getScale();
		const float obbArea = 2.f * (unitBoxScale.x * unitBoxScale.y + unitBoxScale.x * unitBoxScale.z + unitBoxScale.y * unitBoxScale.z);

		return nbl::core::dot(radiance, rec709LumaCoeffs).x * unitHemisphereArea * obbArea;
	}
#endif

	mat4x3 obb; // needs row_major qualifier
	/** TODO new and improved
	mat2x3 obb_base;
	uvec2 radianceRemainder;
	vec3 offset;
	float obb_height;
	**/
};



//
struct StaticViewData_t
{
	uvec2   imageDimensions;
#ifdef __cplusplus
	uint8_t pathDepth;
	uint8_t noRussianRouletteDepth;
	uint16_t samplesPerPixelPerDispatch;
#else
	uint    pathDepth_noRussianRouletteDepth_samplesPerPixelPerDispatch;
#endif
	uint	lightCount;
};

struct RaytraceShaderCommonData_t
{
	mat4 	viewProjMatrixInverse;
	vec3	camPos;
	float   rcpFramesDispatched;
	uint	samplesComputed;
	uint	depth; // 0 if path tracing disabled
	uint	rayCountWriteIx;
	float	textureFootprintFactor;
};

#endif