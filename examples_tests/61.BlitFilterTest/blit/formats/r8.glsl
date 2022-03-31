#ifndef _NBL_GLSL_BLIT_R8_INCLUDED_
#define _NBL_GLSL_BLIT_R8_INCLUDED_

#ifndef _NBL_GLSL_BLIT_DIM_COUNT_
#error _NBL_GLSL_BLIT_DIM_COUNT_ must be defined
#endif

#ifndef _NBL_GLSL_BLIT_OUT_DESCRIPTOR_DEFINED_
#error _NBL_GLSL_BLIT_OUT_DESCRIPTOR_DEFINED_ must be defined
#endif

void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t value, in ivec3 coord)
{
#if NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 1)
#define COORD coord.x
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 2)
#define COORD coord.xy
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 3)
#define COORD coord
#else
#error _NBL_GLSL_BLIT_DIM_COUNT_ not supported
#endif

	const uint encoded = packUnorm4x8(vec4(value.data.r, 0.f, 0.f, 0.f));

	imageStore(_NBL_GLSL_BLIT_OUT_DESCRIPTOR_DEFINED_, COORD, uvec4(encoded, 0, 0, 0));

#undef COORD
}

#endif