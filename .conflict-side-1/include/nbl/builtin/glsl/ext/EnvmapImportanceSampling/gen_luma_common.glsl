#ifndef _NBL_GLSL_EXT_ENVMAP_SAMPLING_GEN_LUMA_COMMON_INCLUDED_
#define _NBL_GLSL_EXT_ENVMAP_SAMPLING_GEN_LUMA_COMMON_INCLUDED_

	
#include <nbl/builtin/glsl/math/functions.glsl>
#include <nbl/builtin/glsl/ext/EnvmapImportanceSampling/structs.glsl>


layout(local_size_x = LUMA_MIP_MAP_GEN_WORKGROUP_DIM, local_size_y = LUMA_MIP_MAP_GEN_WORKGROUP_DIM) in;

layout(push_constant) uniform PushConstants
{
	nbl_glsl_ext_EnvmapSampling_LumaGenShaderData_t data;
} pc;


layout(set = 0, binding = 0) uniform sampler2D envMap;


void consume(in ivec2 pixelCoord, in vec2 uv, in float luma);

void main()
{	
	if (all(lessThan(gl_GlobalInvocationID.xy,pc.data.lumaMapResolution)))
	{
		const ivec2 pixelCoord = ivec2(gl_GlobalInvocationID.xy);
		const vec2 uv = (vec2(pixelCoord)+vec2(0.5))/vec2(pc.data.lumaMapResolution);
		// alpha is not 1.0 in all exr images but we need it to be 1 for correct calculations
		const vec3 envMapSample = textureLod(envMap,uv,0.f).rgb;
		const float luma = dot(vec4(envMapSample,1.f),pc.data.luminanceScales)*sin(nbl_glsl_PI*uv.y);
		//
		consume(pixelCoord,uv,luma);
	}
}

#endif
