#ifndef _NBL_GLSL_BLIT_RG16_INCLUDED_
#define _NBL_GLSL_BLIT_RG16_INCLUDED_

uvec4 nbl_glsl_blit_formats_encode(in vec4 value)
{
	const uint encoded = packUnorm2x16(value.rg);
	return uvec4(encoded, 0, 0, 0);
}

#endif