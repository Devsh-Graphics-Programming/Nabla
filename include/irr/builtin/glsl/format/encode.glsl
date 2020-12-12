#ifndef _IRR_BUILTIN_GLSL_FORMAT_ENCODE_INCLUDED_
#define _IRR_BUILTIN_GLSL_FORMAT_ENCODE_INCLUDED_

#include <irr/builtin/glsl/format/constants.glsl>

uvec2 irr_glsl_encodeRGB19E7(in vec4 col)
{
	int exp = int(bitfieldExtract(x.y, 3*irr_glsl_RGB19E7_MANTISSA_BITS-32, irr_glsl_RGB19E7_EXPONENT_BITS) - irr_glsl_RGB19E7_EXP_BIAS - irr_glsl_RGB19E7_MANTISSA_BITS);
	float scale = exp2(float(exp));//uintBitsToFloat((uint(exp)+127u)<<23u)
	
	vec3 v;
	v.x = int(bitfieldExtract(x.x, 0, irr_glsl_RGB19E7_MANTISSA_BITS))*scale;
	v.y = int(
		bitfieldExtract(x.x, irr_glsl_RGB19E7_MANTISSA_BITS, 32-irr_glsl_RGB19E7_MANTISSA_BITS) | 
		(bitfieldExtract(x.y, 0, irr_glsl_RGB19E7_MANTISSA_BITS-(32-irr_glsl_RGB19E7_MANTISSA_BITS))<<(32-irr_glsl_RGB19E7_MANTISSA_BITS))
	) * scale;
	v.z = int(bitfieldExtract(x.y, irr_glsl_RGB19E7_MANTISSA_BITS-(32-irr_glsl_RGB19E7_MANTISSA_BITS), irr_glsl_RGB19E7_MANTISSA_BITS)) * scale;
	
	return v;
}

uint irr_glsl_encodeRGB10A2(in vec4 col)
{
	const uvec3 rgbMask = uvec3(0x3ffu);
	const vec4 clamped = clamp(col,vec4(0.0),vec4(1.0));
	uvec4 quantized = uvec4(clamped*vec4(vec3(rgbMask),3.0));
	quantized.gba <<= uvec3(10,20,30);
	return quantized.r|quantized.g|quantized.b|quantized.a;
}

#endif