
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_BLIT_FORMATS_ENCODE_INCLUDED_
#define _NBL_HLSL_BLIT_FORMATS_ENCODE_INCLUDED_

#include <nbl/builtin/hlsl/limits/numeric.hlsl>
#include <nbl/builtin/hlsl/format/encode.hlsl>


namespace nbl
{
namespace hlsl
{
namespace blit
{


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

#ifdef SOFTWARE_ENCODE_FORMAT_

uit4 encode(in float4 value)
{
	#if SOFTWARE_ENCODE_FORMAT_ == EF_R32G32_SFLOAT
		return uint4(asuint(value.r), asuint(value.g), 0u, 0u);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R16G16_SFLOAT
		const uint encoded = packHalf2x16(value.rg);
		return uint4(encoded, 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_B10G11R11_UFLOAT_PACK32
		return uint4(nbl_glsl_encodeR11G11B10(value), 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R16_SFLOAT
		const uint encoded = packHalf2x16(float2(value.r, 0.f)).x;
		return uint4(encoded, 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R16G16B16A16_UNORM
		const uint2 encoded = uint2(packUnorm2x16(value.rg), packUnorm2x16(value.ba));
		return uint4(encoded, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_A2B10G10R10_UNORM_PACK32
		const uint encoded = nbl_glsl_encodeRGB10A2_UNORM(value);
		return uint4(encoded, 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R16G16_UNORM
		const uint encoded = packUnorm2x16(value.rg);
		return uint4(encoded, 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R8G8_UNORM
		const uint encoded = packUnorm4x8(float4(value.rg, 0.f, 0.f));
		return uint4(encoded, 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R16_UNORM
		const uint encoded = packUnorm2x16(float2(value.r, 0.f));
		return uint4(encoded, 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R8_UNORM
		const uint encoded = packUnorm4x8(float4(value.r, 0.f, 0.f, 0.f));
		return uint4(encoded, 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R16G16B16A16_SNORM
		const uint2 encoded = uint2(packSnorm2x16(value.rg), packSnorm2x16(value.ba));
		return uint4(encoded, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R16G16_SNORM
		const uint encoded = packSnorm2x16(value.rg);
		return uint4(encoded, 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R8G8_SNORM
		const uint encoded = packSnorm4x8(float4(value.rg, 0.f, 0.f));
		return uint4(encoded, 0, 0, 0);
	#elif SOFTWARE_ENCODE_FORMAT_ == EF_R16_SNORM
		const uint encoded = packSnorm2x16(float2(value.r, 0.f));
		return uint4(encoded, 0, 0, 0);
	#else
		#error "Software Encode of this format not implemented or possible!"
	#endif
}
#endif


}
}
}
#endif