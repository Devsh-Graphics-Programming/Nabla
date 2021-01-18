#ifndef _COMMON_INCLUDED_
#define _COMMON_INCLUDED_

#define MAX_ACCUMULATED_SAMPLES (1024*1024)

#define WORKGROUP_SIZE 256

#ifdef __cplusplus
	#define uint uint32_t
	struct uvec2
	{
		uint32_t x,y;
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

#endif
