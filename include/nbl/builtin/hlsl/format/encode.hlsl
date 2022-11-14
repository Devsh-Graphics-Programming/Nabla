
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_FORMAT_ENCODE_INCLUDED_
#define _NBL_BUILTIN_HLSL_FORMAT_ENCODE_INCLUDED_

#include <nbl/builtin/hlsl/format/constants.hlsl>
#include <nbl/builtin/hlsl/limits/numeric.hlsl>

namespace nbl
{
namespace hlsl
{
namespace format
{


uint3 impl_sharedExponentEncodeCommon(in float3 clamped, in int newExpBias, in int newMaxExp, in int mantissaBits, out int shared_exp)
{
	const float maxrgb = max(max(clamped.r,clamped.g),clamped.b);
	// TODO: optimize this
	const int f32_exp = int(ieee754::extract_biased_exponent(maxrgb))-126;

	shared_exp = clamp(f32_exp,-newExpBias,newMaxExp+1);
	
	float scale = exp2(mantissaBits-shared_exp);
	const uint maxm = uint(maxrgb*scale + 0.5);
	const bool need = maxm==(0x1u<<mantissaBits);
	scale = select(need, 0.5 * scale, scale);
	shared_exp = select(need, shared_exp + 1, shared_exp);
	return uint3(clamped*scale + float3(0.5));
}

uint2 encodeRGB19E7(in float3 col)
{
	const float3 clamped = clamp(col,float3(0.0),float3(MAX_RGB19E7));

	int shared_exp;
	const uint3 mantissas = impl_sharedExponentEncodeCommon(clamped,RGB19E7_EXP_BIAS,MAX_RGB19E7_EXP,RGB19E7_MANTISSA_BITS,shared_exp);

	uint2 encoded;
	encoded.x = bitfieldInsert(mantissas.x,mantissas.y,RGB19E7_COMPONENT_BITOFFSETS[1],RGB19E7_G_COMPONENT_SPLIT);
	encoded.y = bitfieldInsert(
		mantissas.y>>RGB19E7_G_COMPONENT_SPLIT,
		mantissas.z,
		RGB19E7_COMPONENT_BITOFFSETS[2],
		RGB19E7_MANTISSA_BITS)
	| uint((shared_exp+RGB19E7_EXP_BIAS)<<RGB19E7_COMPONENT_BITOFFSETS[3]);

	return encoded;
}

uint2 encodeRGB18E7S3(in float3 col)
{
	const float3 absCol = abs(col);
	const float3 clamped = min(absCol,float3(MAX_RGB18E7S3));

	int shared_exp;
	const uint3 mantissas = impl_sharedExponentEncodeCommon(clamped,RGB18E7S3_EXP_BIAS,MAX_RGB18E7S3_EXP,RGB18E7S3_MANTISSA_BITS,shared_exp);

	uint3 signs = floatBitsToUint(col)&0x80000000u;
	signs.xy >>= uint2(2u,1u);

	uint2 encoded;
	encoded.x = bitfieldInsert(mantissas.x,mantissas.y,RGB18E7S3_COMPONENT_BITOFFSETS[1],RGB18E7S3_G_COMPONENT_SPLIT);
	encoded.y = bitfieldInsert(
		mantissas.y>>RGB18E7S3_G_COMPONENT_SPLIT,
		mantissas.z,
		RGB18E7S3_COMPONENT_BITOFFSETS[2],
		RGB18E7S3_MANTISSA_BITS)
	| uint((shared_exp+RGB18E7S3_EXP_BIAS)<<RGB18E7S3_COMPONENT_BITOFFSETS[3])
	| signs.x | signs.y | signs.z;

	return encoded;
}

//
uint encodeRGB10A2_UNORM(in float4 col)
{
	const uint3 rgbMask = uint3(0x3ffu);
	const float4 clamped = clamp(col,float4(0.0),float4(1.0));
	uint4 quantized = uint4(clamped*float4(float3(rgbMask),3.0));
	quantized.gba <<= uint3(10,20,30);
	return quantized.r|quantized.g|quantized.b|quantized.a;
}
uint encodeRGB10A2_SNORM(in float4 col)
{
	const int4 mask = int4(int3(0x3ffu),0x3u);
	const uint3 halfMask = uint3(0x1ffu);
	const float4 clamped = clamp(col,float4(-1.f),float4(1.f));
	uint4 quantized = uint4(int4(clamped.rgb*float3(halfMask),int(clamped.a))&mask);
	quantized.gba <<= uint3(10,20,30);
	return quantized.r|quantized.g|quantized.b|quantized.a;
}

// TODO: break it down into uint encode_ufloat_exponent(in float _f32) and encode_ufloat_mantissa(in float _f32, in uint mantissaBits, in bool leadingOne)
uint encode_ufloat(in float _f32, in uint mantissaBits, in uint expBits)
{
	uint minSinglePrecisionVal = floatBitsToUint(ieee754::min(expBits, mantissaBits));
	uint maxSinglePrecisionVal = floatBitsToUint(ieee754::max(expBits, mantissaBits));

	if (_f32 < uintBitsToFloat(maxSinglePrecisionVal))
	{
		if (_f32 < uintBitsToFloat(minSinglePrecisionVal))
			return 0;

		const int exp = ieee754::extract_exponent(_f32);
		const uint mantissa = ieee754::extract_mantissa(_f32);

		const uint encodedValue = ieee754::encode_ufloat_impl(exp, expBits, mantissa, mantissaBits);

		return encodedValue;
	}

	const uint expMask = ieee754::compute_exponent_mask(expBits, mantissaBits);
	const uint mantissaMask = ieee754::compute_mantissa_mask(mantissaBits);

	return expMask | ( select(isnan(_f32), mantissaMask, 0u) );
}

uint to11bitFloat(in float _f32)
{
	const uint mantissaBits = 6;
	return encode_ufloat(_f32, mantissaBits, 11-mantissaBits);
}

uint to10bitFloat(in float _f32)
{
	const uint mantissaBits = 5;
	return encode_ufloat(_f32, mantissaBits, 10-mantissaBits);
}

uint encodeR11G11B10(in float4 col)
{
	const uint r = to11bitFloat(col.r);
	const uint g = to11bitFloat(col.g) << 11;
	const uint b = to10bitFloat(col.b) << 22;
	const uint encoded = b | g | r;
	return encoded;
}


}
}
}

#endif