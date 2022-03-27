#ifndef _NBL_GLSL_BLIT_QUAD_CHANNEL_REQUIRED_FORMATS_INCLUDED_
#define _NBL_GLSL_BLIT_QUAD_CHANNEL_REQUIRED_FORMATS_INCLUDED_

#if 0
void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t value, in ivec3 coord)
{
	imageStore(outImage, coord, value);
}

void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t value, in ivec2 coord)
{
	imageStore(outImage, coord, value);
}
#endif

void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t value, in int coord)
{
	imageStore(outImage, coord, value.data);
}

#endif