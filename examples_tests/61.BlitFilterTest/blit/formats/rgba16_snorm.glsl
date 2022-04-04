#ifndef _NBL_GLSL_BLIT_RGBA16_SNORM_INCLUDED_
#define _NBL_GLSL_BLIT_RGBA16_SNORM_INCLUDED_

uvec4 nbl_glsl_blit_formats_encode(in vec4 value)
{
	const uvec2 encoded = uvec2(packSnorm2x16(value.rg), packSnorm2x16(value.ba));
	return uvec4(encoded, 0, 0);
}

vec4 nbl_glsl_blit_formats_decode(in uvec4 value)
{
	return vec4(unpackSnorm2x16(value.x), unpackSnorm2x16(value.y));
}

#endif