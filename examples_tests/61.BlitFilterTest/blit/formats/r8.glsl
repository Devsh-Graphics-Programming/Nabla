#ifndef _NBL_GLSL_BLIT_R8_INCLUDED_
#define _NBL_GLSL_BLIT_R8_INCLUDED_

uvec4 nbl_glsl_blit_formats_encode(in vec4 value)
{
	const uint encoded = packUnorm4x8(vec4(value.r, 0.f, 0.f, 0.f));
	return uvec4(encoded, 0, 0, 0);
}

#endif