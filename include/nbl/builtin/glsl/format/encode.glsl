#ifndef _IRR_BUILTIN_GLSL_FORMAT_ENCODE_INCLUDED_
#define _IRR_BUILTIN_GLSL_FORMAT_ENCODE_INCLUDED_

#include <nbl/builtin/glsl/format/constants.glsl>
#include <nbl/builtin/glsl/limits/numeric.glsl>

uvec3 nbl_glsl_impl_sharedExponentEncodeCommon(in vec3 clamped, in int newExpBias, in int newMaxExp, in int mantissaBits, out int shared_exp)
{
	const float maxrgb = max(max(clamped.r,clamped.g),clamped.b);
	// TODO: optimize this
	const int f32_exp = int(nbl_glsl_ieee754_extract_biased_exponent(maxrgb))-126;

	shared_exp = clamp(f32_exp,-newExpBias,newMaxExp+1);
	
	float scale = exp2(mantissaBits-shared_exp);
	const uint maxm = uint(maxrgb*scale + 0.5);
	const bool need = maxm==(0x1u<<mantissaBits);
	scale = need ? 0.5*scale:scale;
	shared_exp = need ? (shared_exp+1):shared_exp;
	return uvec3(clamped*scale + vec3(0.5));
}

uvec2 nbl_glsl_encodeRGB19E7(in vec3 col)
{
	const vec3 clamped = clamp(col,vec3(0.0),vec3(nbl_glsl_MAX_RGB19E7));

	int shared_exp;
	const uvec3 mantissas = nbl_glsl_impl_sharedExponentEncodeCommon(clamped,nbl_glsl_RGB19E7_EXP_BIAS,nbl_glsl_MAX_RGB19E7_EXP,nbl_glsl_RGB19E7_MANTISSA_BITS,shared_exp);

	uvec2 encoded;
	encoded.x = bitfieldInsert(mantissas.x,mantissas.y,nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[1],nbl_glsl_RGB19E7_G_COMPONENT_SPLIT);
	encoded.y = bitfieldInsert(
		mantissas.y>>nbl_glsl_RGB19E7_G_COMPONENT_SPLIT,
		mantissas.z,
		nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[2],
		nbl_glsl_RGB19E7_MANTISSA_BITS)
	| uint((shared_exp+nbl_glsl_RGB19E7_EXP_BIAS)<<nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[3]);

	return encoded;
}

uvec2 nbl_glsl_encodeRGB18E7S3(in vec3 col)
{
	const vec3 absCol = abs(col);
	const vec3 clamped = min(absCol,vec3(nbl_glsl_MAX_RGB18E7S3));

	int shared_exp;
	const uvec3 mantissas = nbl_glsl_impl_sharedExponentEncodeCommon(clamped,nbl_glsl_RGB18E7S3_EXP_BIAS,nbl_glsl_MAX_RGB18E7S3_EXP,nbl_glsl_RGB18E7S3_MANTISSA_BITS,shared_exp);

	uvec3 signs = floatBitsToUint(col)&0x80000000u;
	signs.xy >>= uvec2(2u,1u);

	uvec2 encoded;
	encoded.x = bitfieldInsert(mantissas.x,mantissas.y,nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS[1],nbl_glsl_RGB18E7S3_G_COMPONENT_SPLIT);
	encoded.y = bitfieldInsert(
		mantissas.y>>nbl_glsl_RGB18E7S3_G_COMPONENT_SPLIT,
		mantissas.z,
		nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS[2],
		nbl_glsl_RGB18E7S3_MANTISSA_BITS)
	| uint((shared_exp+nbl_glsl_RGB18E7S3_EXP_BIAS)<<nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS[3])
	| signs.x | signs.y | signs.z;

	return encoded;
}

//
uint nbl_glsl_encodeRGB10A2_UNORM(in vec4 col)
{
	const uvec3 rgbMask = uvec3(0x3ffu);
	const vec4 clamped = clamp(col,vec4(0.0),vec4(1.0));
	uvec4 quantized = uvec4(clamped*vec4(vec3(rgbMask),3.0));
	quantized.gba <<= uvec3(10,20,30);
	return quantized.r|quantized.g|quantized.b|quantized.a;
}
uint nbl_glsl_encodeRGB10A2_SNORM(in vec4 col)
{
	const ivec4 mask = ivec4(ivec3(0x3ffu),0x3u);
	const uvec3 halfMask = uvec3(0x1ffu);
	const vec4 clamped = clamp(col,vec4(-1.f),vec4(1.f));
	uvec4 quantized = uvec4(ivec4(clamped.rgb*vec3(halfMask),int(clamped.a))&mask);
	quantized.gba <<= uvec3(10,20,30);
	return quantized.r|quantized.g|quantized.b|quantized.a;
}

// TODO: break it down into uint nbl_glsl_encode_ufloat_exponent(in float _f32) and nbl_glsl_encode_ufloat_mantissa(in float _f32, in uint mantissaBits, in bool leadingOne)
uint nbl_glsl_encode_ufloat(in float _f32, in uint mantissaBits)
{
	uint minSinglePrecisionVal = floatBitsToUint(6.10 * 1e-5);
	uint maxSinglePrecisionVal = floatBitsToUint(6.50 * 1e4);

	if (_f32 < uintBitsToFloat(maxSinglePrecisionVal))
	{
		if (_f32 < uintBitsToFloat(minSinglePrecisionVal))
			return 0;

		const int exp = int(nbl_glsl_ieee754_extract_biased_exponent(_f32) - 127);
		const uint mantissa = nbl_glsl_ieee754_extract_mantissa(_f32);
		const uint bias = 15;

		const uint e = uint(exp + bias);
		const uint m = mantissa >> (23 - mantissaBits);
		const uint encodedValue = (e << mantissaBits) | m;

		return encodedValue;
	}

	const uint expMask = 0x1fu << mantissaBits;
	const uint mantissaMask = nbl_glsl_ieee754_compute_mantissa_mask(mantissaBits);

	return expMask | (isnan(_f32) ? mantissaMask : 0u);
}

uint to11bitFloat(in float _f32)
{
	const uint mantissaBits = 6;
	return nbl_glsl_encode_ufloat(_f32, mantissaBits);
}

uint to10bitFloat(in float _f32)
{
	const uint mantissaBits = 5;
	return nbl_glsl_encode_ufloat(_f32, mantissaBits);
}

uint nbl_glsl_encodeR11G11B10(in vec4 col)
{
	const uint r = to11bitFloat(col.r);
	const uint g = to11bitFloat(col.g) << 11;
	const uint b = to10bitFloat(col.b) << 22;
	const uint encoded = b | g | r;
	return encoded;
}
#endif