#ifndef _RAYGEN_COMMON_INCLUDED_
#define _RAYGEN_COMMON_INCLUDED_

#include "common.glsl"
#if WORKGROUP_SIZE!=256
	#error "Hardcoded 16 should be NBL_SQRT(WORKGROUP_SIZE)"
#endif
layout(local_size_x = 16, local_size_y = 16) in;

//
layout(set = 1, binding = 0, rgba32f) restrict uniform image2D framebuffer;
#include <irr/builtin/glsl/ext/RadeonRays/ray.glsl>
layout(set = 1, binding = 1, std430) restrict writeonly buffer Rays
{
	irr_glsl_ext_RadeonRays_ray rays[];
};
// lights
layout(set = 1, binding = 2, std430) restrict readonly buffer CumulativeLightPDF
{
	uint lightCDF[];
};
layout(set = 1, binding = 3, std430, row_major) restrict readonly buffer Lights
{
	SLight light[];
};
layout(set = 1, binding = 4, std430, row_major) restrict readonly buffer LightRadiances
{
	uvec2 lightRadiance[]; // Watts / steriadian / steradian in rgb19e7
};

struct RaygenShaderData_t
{
    RaytraceShaderCommonData_t common;
    mat4x3  frustumCorners;
    mat4x3  normalMatrixAndCameraPos;
    vec2    rcpPixelSize;
    vec2    rcpHalfPixelSize;
    float   depthLinearizationConstant;
    uint    samplesComputedPerPixel;
    uint    padding[2];
};

struct ResolveShaderData_t
{
    RaytraceShaderCommonData_t common;
    uint    framesDone;
    float   rcpFramesDone;
    uint    padding[2];
};

#endif
