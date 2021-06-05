#ifndef _RAYTRACE_COMMON_GLSL_INCLUDED_
#define _RAYTRACE_COMMON_GLSL_INCLUDED_


#include "raytraceCommon.h"


#extension GL_EXT_shader_16bit_storage : require
layout(local_size_x = WORKGROUP_DIM, local_size_y = WORKGROUP_DIM) in;

#include "virtualGeometry.glsl"

// lights
layout(set = 1, binding = 4, std430) restrict readonly buffer CumulativeLightPDF
{
	uint lightCDF[];
};
layout(set = 1, binding = 5, std430, row_major) restrict readonly buffer Lights
{
	SLight light[];
};
layout(set = 1, binding = 6, std430, row_major) restrict readonly buffer LightRadiances
{
	uvec2 lightRadiance[]; // Watts / steriadian / steradian in rgb19e7
};


layout(set = 2, binding = 0, row_major) uniform StaticViewData
{
	StaticViewData_t staticViewData;
};
layout(set = 2, binding = 1, rg32ui) restrict uniform uimage2DArray accumulation;
#include <nbl/builtin/glsl/ext/RadeonRays/ray.glsl>
layout(set = 2, binding = 2, std430) restrict buffer Rays
{
	nbl_glsl_ext_RadeonRays_ray rays[];
};
#include <nbl/builtin/glsl/format/decode.glsl>
#include <nbl/builtin/glsl/format/encode.glsl>
vec3 fetchAccumulation(in uvec2 coord, in uint subsample)
{
    const uvec2 data = imageLoad(accumulation,ivec3(coord,subsample)).rg;
	return nbl_glsl_decodeRGB19E7(data);
}
void storeAccumulation(in vec3 color, in uvec2 coord, in uint subsample)
{
	const uvec2 data = nbl_glsl_encodeRGB19E7(color);
	imageStore(accumulation,ivec3(coord,subsample),uvec4(data,0u,0u));
}



layout(push_constant, row_major) uniform PushConstants
{
	RaytraceShaderCommonData_t cummon;
} pc;


#endif
