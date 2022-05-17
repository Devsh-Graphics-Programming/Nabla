#ifndef _NBL_GLSL_BLIT_R16_SNORM_INCLUDED_
#define _NBL_GLSL_BLIT_R16_SNORM_INCLUDED_

uvec4 nbl_glsl_blit_formats_encode(in vec4 value)
{
	const uint encoded = packSnorm2x16(vec2(value.r, 0.f));
	return uvec4(encoded, 0, 0, 0);
}

#endif