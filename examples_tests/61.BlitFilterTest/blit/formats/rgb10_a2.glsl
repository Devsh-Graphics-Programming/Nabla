#ifndef _NBL_GLSL_BLIT_RGB10_A2_INCLUDED_
#define _NBL_GLSL_BLIT_RGB10_A2_INCLUDED_

#include <nbl/builtin/glsl/limits/numeric.glsl> // Todo(achal): Can remove this after merging with `master`
#include <nbl/builtin/glsl/format/encode.glsl>
#include <nbl/builtin/glsl/format/decode.glsl>

uvec4 nbl_glsl_blit_formats_encode(in vec4 value)
{
	const uint encoded = nbl_glsl_encodeRGB10A2_UNORM(value);
	return uvec4(encoded, 0, 0, 0);
}

vec4 nbl_glsl_blit_formats_decode(in uvec4 value)
{
	return nbl_glsl_decodeRGB10A2_UNORM(value.x);
}

#endif