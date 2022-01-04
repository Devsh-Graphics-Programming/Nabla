#ifndef _COMMON_INCLUDED_
#define _COMMON_INCLUDED_


#define RAYCOUNT_N_BUFFERING 4
#define RAYCOUNT_N_BUFFERING_MASK (RAYCOUNT_N_BUFFERING-1)

#define MAX_TRIANGLES_IN_BATCH 16384

//! Brief guideline to good path depth limits
// Want to see stuff with indirect lighting on the other side of a pane of glass
// 5 = glass frontface->glass backface->diffuse surface->diffuse surface->light
// Want to see through a glass box, vase, or office 
// 7 = glass frontface->glass backface->glass frontface->glass backface->diffuse surface->diffuse surface->light
// pick higher numbers for better GI and less bias
#define MAX_PATH_DEPTH 8
// need to bump to 2 in case of NEE + MIS, 3 in case of Path Guiding
#define SAMPLING_STRATEGY_COUNT 1
// One less because the first vertex is rasterized
#define QUANTIZED_DIMENSIONS_PER_SAMPLE ((MAX_PATH_DEPTH-1)*SAMPLING_STRATEGY_COUNT)


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
