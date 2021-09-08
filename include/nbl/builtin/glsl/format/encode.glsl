#ifndef _IRR_BUILTIN_GLSL_FORMAT_ENCODE_INCLUDED_
#define _IRR_BUILTIN_GLSL_FORMAT_ENCODE_INCLUDED_

#include <nbl/builtin/glsl/format/constants.glsl>

uvec3 nbl_glsl_impl_sharedExponentEncodeCommon(in vec3 clamped, in int newExpBias, in int newMaxExp, in int mantissaBits, out int shared_exp)
{
	float maxrgb = max(max(clamped.r,clamped.g),clamped.b);
	// TODO: optimize this
	const int f32_exp = ((floatBitsToInt(maxrgb)>>23) & 0xff) - 126;

	shared_exp = clamp(f32_exp,-newExpBias,newMaxExp+1);
	
	float scale = exp2(mantissaBits-shared_exp);
	const uint maxm = uint(maxrgb*scale + 0.5);
	const bool need = maxm==(0x1u<<mantissaBits);
	scale = need ? 0.5*scale:scale;
	shared_exp = need ? (shared_exp+1):shared_exp;
	const uvec3 mantissas = uvec3(clamped*scale + vec3(0.5));
	return mantissas;
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

uint nbl_glsl_encodeRGB10A2(in vec4 col)
{
	const uvec3 rgbMask = uvec3(0x3ffu);
	const vec4 clamped = clamp(col,vec4(0.0),vec4(1.0));
	uvec4 quantized = uvec4(clamped*vec4(vec3(rgbMask),3.0));
	quantized.gba <<= uvec3(10,20,30);
	return quantized.r|quantized.g|quantized.b|quantized.a;
}

#endif