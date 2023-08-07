#ifndef _NBL_GLSL_EXT_ENVMAP_SAMPLING_PARAMETERS_STRUCT_INCLUDED_
#define _NBL_GLSL_EXT_ENVMAP_SAMPLING_PARAMETERS_STRUCT_INCLUDED_

#ifdef __cplusplus
	#define uint uint32_t
	struct uvec2
	{
		uint x,y;
	};
	struct vec2
	{
		float x,y;
	};
	struct vec3
	{
		float x,y,z;
	};
	#define vec4 nbl::core::vectorSIMDf
	#define mat4 nbl::core::matrix4SIMD
	#define mat4x3 nbl::core::matrix3x4SIMD
#endif
	
struct LumaMipMapGenShaderData_t
{
    vec4 luminanceScales;
    uint calcLuma;
    uint sinFactor;
    vec2 padding;
};

struct WarpMapGenShaderData_t
{
    uint lumaMipCount;
    vec3 padding;
};

#endif
