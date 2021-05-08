#ifndef _RAYGEN_COMMON_INCLUDED_
#define _RAYGEN_COMMON_INCLUDED_

#include "common.glsl"
#if WORKGROUP_SIZE!=256
	#error "Hardcoded 16 should be NBL_SQRT(WORKGROUP_SIZE)"
#endif
#define WORKGROUP_DIM 16
#ifndef __cplusplus
layout(local_size_x = WORKGROUP_DIM, local_size_y = WORKGROUP_DIM) in;
#endif


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
};



//
struct StaticViewData_t
{
	vec3	envmapBaseColor;
	uint	lightCount;
	vec2    rcpPixelSize;
	vec2    rcpHalfPixelSize;
	uvec2   imageDimensions;
	uint    samplesPerPixelPerDispatch;
	uint    samplesPerRowPerDispatch;
};

struct RaytraceShaderCommonData_t
{
	mat4	inverseMVP;
	mat4x3  ndcToV;
	uint    samplesComputedPerPixel;
	uint    framesDispatched;
    float   rcpFramesDispatched;
	float	padding0;
};


#ifndef __cplusplus
layout(push_constant, row_major) uniform PushConstants
{
	RaytraceShaderCommonData_t cummon; 
} pc;
layout(set = 1, binding = 0, row_major) uniform StaticViewData
{
	StaticViewData_t staticViewData;
};
layout(set = 1, binding = 1, rg32ui) restrict uniform uimage2D accumulation;
#include <nbl/builtin/glsl/ext/RadeonRays/ray.glsl>
layout(set = 1, binding = 2, std430) restrict buffer Rays
{
	nbl_glsl_ext_RadeonRays_ray rays[];
};
// lights
layout(set = 1, binding = 3, std430) restrict readonly buffer CumulativeLightPDF
{
	uint lightCDF[];
};
layout(set = 1, binding = 4, std430, row_major) restrict readonly buffer Lights
{
	SLight light[];
};
layout(set = 1, binding = 5, std430, row_major) restrict readonly buffer LightRadiances
{
	uvec2 lightRadiance[]; // Watts / steriadian / steradian in rgb19e7
};

#include <nbl/builtin/glsl/format/decode.glsl>
#include <nbl/builtin/glsl/format/encode.glsl>
vec3 fetchAccumulation(in ivec2 coord)
{
    const uvec2 data = imageLoad(accumulation,coord).rg;
	return nbl_glsl_decodeRGB19E7(data);
}
void storeAccumulation(in vec3 color, in ivec2 coord)
{
	const uvec2 data = nbl_glsl_encodeRGB19E7(color);
	imageStore(accumulation,coord,uvec4(data,0u,0u));
}

#endif

#endif