#ifndef _RAYGEN_COMMON_INCLUDED_
#define _RAYGEN_COMMON_INCLUDED_

#include "common.glsl"
#if WORKGROUP_SIZE!=256
	#error "Hardcoded 16 should be NBL_SQRT(WORKGROUP_SIZE)"
#endif
#ifndef __cplusplus
#define WORKGROUP_DIM 16
layout(local_size_x = WORKGROUP_DIM, local_size_y = WORKGROUP_DIM) in;
#endif


struct SLight
{
#ifdef __cplusplus
	SLight() : obb() {}
	SLight(const SLight& other) : obb(other.obb) {}
	SLight(const irr::core::aabbox3df& bbox, const irr::core::matrix3x4SIMD& tform) : SLight()
	{
		auto extent = bbox.getExtent();
		obb.setScale(irr::core::vectorSIMDf(extent.X, extent.Y, extent.Z));
		obb.setTranslation(irr::core::vectorSIMDf(bbox.MinEdge.X, bbox.MinEdge.Y, bbox.MinEdge.Z));

		obb = irr::core::concatenateBFollowedByA(tform, obb);
	}

	inline SLight& operator=(SLight&& other) noexcept
	{
		std::swap(obb, other.obb);

		return *this;
	}

	// also known as an upper bound on lumens put into the scene
	inline float computeFluxBound(const irr::core::vectorSIMDf& radiance) const
	{
		const irr::core::vectorSIMDf rec709LumaCoeffs(0.2126f, 0.7152f, 0.0722f, 0.f);
		const auto unitHemisphereArea = 2.f * irr::core::PI<float>();

		const auto unitBoxScale = obb.getScale();
		const float obbArea = 2.f * (unitBoxScale.x * unitBoxScale.y + unitBoxScale.x * unitBoxScale.z + unitBoxScale.y * unitBoxScale.z);

		return irr::core::dot(radiance, rec709LumaCoeffs).x * unitHemisphereArea * obbArea;
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
	mat4x3  frustumCorners;
	mat4x3  normalMatrixAndCameraPos;
	float   depthLinearizationConstant;
	uint    samplesComputedPerPixel;
	uint    framesDispatched;
    float   rcpFramesDispatched;
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
#include <irr/builtin/glsl/ext/RadeonRays/ray.glsl>
layout(set = 1, binding = 2, std430) restrict /*writeonly/readonly TODO depending on stage*/ buffer Rays
{
	irr_glsl_ext_RadeonRays_ray rays[];
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

#include <irr/builtin/glsl/format/decode.glsl>
#include <irr/builtin/glsl/format/encode.glsl>
vec3 fetchAccumulation(in ivec2 coord)
{
    const uvec2 data = imageLoad(accumulation,coord).rg;
    return irr_glsl_decodeRGB19E7(data);
}
void storeAccumulation(in vec3 color, in ivec2 coord)
{
	imageStore(accumulation,coord,uvec4(irr_glsl_encodeRGB19E7(color),0u,0u));
}
#endif


#endif
