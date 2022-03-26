#ifndef _NBL_GLSL_BLIT_SINGLE_CHANNEL_REQUIRED_FORMATS_INCLUDED_
#define _NBL_GLSL_BLIT_SINGLE_CHANNEL_REQUIRED_FORMATS_INCLUDED_

#if 0
void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t value, in ivec3 coord)
{
	imageStore(outImage, coord, vec4(value.data.r, 0.f, 0.f, 0.f));
}

void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t value, in ivec2 coord)
{
	imageStore(outImage, coord, vec4(value.data.r, 0.f, 0.f, 0.f));
}
#endif

void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t value, in int coord)
{
	imageStore(outImage, coord, vec4(value.data.r, 0.f, 0.f, 0.f));
}

#endif