#ifndef _NBL_GLSL_BLIT_RGBA16_INCLUDED_
#define _NBL_GLSL_BLIT_RGBA16_INCLUDED_

uvec4 nbl_glsl_blit_formats_encode(in vec4 value)
{
	const uvec2 encoded = uvec2(packUnorm2x16(value.rg), packUnorm2x16(value.ba));
	return uvec4(encoded, 0, 0);
}

vec4 nbl_glsl_blit_formats_decode(in uvec4 value)
{
	return vec4(unpackUnorm2x16(value.x), unpackUnorm2x16(value.y));
}

#endif