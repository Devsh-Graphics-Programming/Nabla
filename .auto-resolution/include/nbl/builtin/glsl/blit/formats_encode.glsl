#ifndef _NBL_GLSL_BLIT_FORMATS_ENCODE_INCLUDED_
#define _NBL_GLSL_BLIT_FORMATS_ENCODE_INCLUDED_

#include <nbl/builtin/glsl/limits/numeric.glsl>
#include <nbl/builtin/glsl/format/encode.glsl>

#define EF_R32G32_SFLOAT 109
#define EF_R16G16_SFLOAT 89
#define EF_B10G11R11_UFLOAT_PACK32 128
#define EF_R16_SFLOAT 82

#define EF_R16G16B16A16_UNORM 97
#define EF_A2B10G10R10_UNORM_PACK32 70
#define EF_R16G16_UNORM 83
#define EF_R8G8_UNORM 22
#define EF_R16_UNORM 76
#define EF_R8_UNORM 15

#define EF_R16G16B16A16_SNORM 98
#define EF_R16G16_SNORM 84
#define EF_R8G8_SNORM 23
#define EF_R16_SNORM 77

#ifdef _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_

uvec4 nbl_glsl_blit_encode(in vec4 value)
{
#if _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R32G32_SFLOAT
	return uvec4(floatBitsToUint(value.r), floatBitsToUint(value.g), 0u, 0u);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R16G16_SFLOAT
	const uint encoded = packHalf2x16(value.rg);
	return uvec4(encoded, 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_B10G11R11_UFLOAT_PACK32
	return uvec4(nbl_glsl_encodeR11G11B10(value), 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R16_SFLOAT
	const uint encoded = packHalf2x16(vec2(value.r, 0.f)).x;
	return uvec4(encoded, 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R16G16B16A16_UNORM
	const uvec2 encoded = uvec2(packUnorm2x16(value.rg), packUnorm2x16(value.ba));
	return uvec4(encoded, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_A2B10G10R10_UNORM_PACK32
	const uint encoded = nbl_glsl_encodeRGB10A2_UNORM(value);
	return uvec4(encoded, 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R16G16_UNORM
	const uint encoded = packUnorm2x16(value.rg);
	return uvec4(encoded, 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R8G8_UNORM
	const uint encoded = packUnorm4x8(vec4(value.rg, 0.f, 0.f));
	return uvec4(encoded, 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R16_UNORM
	const uint encoded = packUnorm2x16(vec2(value.r, 0.f));
	return uvec4(encoded, 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R8_UNORM
	const uint encoded = packUnorm4x8(vec4(value.r, 0.f, 0.f, 0.f));
	return uvec4(encoded, 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R16G16B16A16_SNORM
	const uvec2 encoded = uvec2(packSnorm2x16(value.rg), packSnorm2x16(value.ba));
	return uvec4(encoded, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R16G16_SNORM
	const uint encoded = packSnorm2x16(value.rg);
	return uvec4(encoded, 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R8G8_SNORM
	const uint encoded = packSnorm4x8(vec4(value.rg, 0.f, 0.f));
	return uvec4(encoded, 0, 0, 0);
#elif _NBL_GLSL_BLIT_SOFTWARE_ENCODE_FORMAT_==EF_R16_SNORM
	const uint encoded = packSnorm2x16(vec2(value.r, 0.f));
	return uvec4(encoded, 0, 0, 0);
#else
	#error "Software Encode of this format not implemented or possible!"
#endif
}
#endif

#endif