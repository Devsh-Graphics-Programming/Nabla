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
#else

#extension GL_EXT_shader_16bit_storage : require

#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute.glsl>
/*
#define _NBL_VG_USE_SSBO
#define _NBL_VG_SSBO_DESCRIPTOR_SET 1
#define _NBL_VG_USE_SSBO_UINT
#define _NBL_VG_SSBO_UINT_BINDING 0
#define _NBL_VG_USE_SSBO_UVEC3
#define _NBL_VG_SSBO_UVEC3_BINDING 1
#define _NBL_VG_USE_SSBO_INDEX
#define _NBL_VG_SSBO_INDEX_BINDING 2
// TODO: remove after all quantization optimizations in CSerializedLoader and the like
#define _NBL_VG_USE_SSBO_UVEC2
#define _NBL_VG_SSBO_UVEC2_BINDING 3
#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute_fetch.glsl>
*/

#endif

#endif
