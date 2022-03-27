#ifndef _NBL_GLSL_BLIT_SINGLE_CHANNEL_REQUIRED_FORMATS_INCLUDED_
#define _NBL_GLSL_BLIT_SINGLE_CHANNEL_REQUIRED_FORMATS_INCLUDED_

#include <nbl/builtin/glsl/macros.glsl>

#ifndef DIM_COUNT
#error DIM_COUNT must be defined
#endif

void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t value, in ivec3 coord)
{
#if NBL_GLSL_EQUAL(DIM_COUNT, 1)
	#define COORD coord.x
#elif NBL_GLSL_EQUAL(DIM_COUNT, 2)
	#define COORD coord.xy
#elif NBL_GLSL_EQUAL(DIM_COUNT, 3)
	#define COORD coord
#else
	#error DIM_COUNT not supported
#endif
	imageStore(outImage, coord, value.data);
}

#endif