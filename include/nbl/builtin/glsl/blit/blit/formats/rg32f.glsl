#ifndef _NBL_GLSL_BLIT_RG32F_INCLUDED_
#define _NBL_GLSL_BLIT_RG32F_INCLUDED_

uvec4 nbl_glsl_blit_formats_encode(in vec4 value)
{
	const uvec4 encoded = uvec4(floatBitsToUint(value.r), floatBitsToUint(value.g), 0u, 0u);
	return encoded;
}

#endif