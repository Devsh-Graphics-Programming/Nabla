#ifndef _COMMON_INCLUDED_
#define _COMMON_INCLUDED_


#define RAYCOUNT_N_BUFFERING 4
#define RAYCOUNT_N_BUFFERING_MASK (RAYCOUNT_N_BUFFERING-1)

#define MAX_TRIANGLES_IN_BATCH 16384
// could be 0x200000
#define MAX_ACCUMULATED_SAMPLES 0x10000


#define WORKGROUP_SIZE 256


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


#endif
