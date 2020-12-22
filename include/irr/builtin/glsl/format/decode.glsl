#ifndef _IRR_BUILTIN_GLSL_FORMAT_DECODE_INCLUDED_
#define _IRR_BUILTIN_GLSL_FORMAT_DECODE_INCLUDED_

#include <irr/builtin/glsl/format/constants.glsl>

vec3 irr_glsl_decodeRGB19E7(in uvec2 x)
{
	int exp = int(bitfieldExtract(x[irr_glsl_RGB19E7_COMPONENT_INDICES[3]], irr_glsl_RGB19E7_COMPONENT_BITOFFSETS[3], irr_glsl_RGB19E7_EXPONENT_BITS) - irr_glsl_RGB19E7_EXP_BIAS - irr_glsl_RGB19E7_MANTISSA_BITS);
	float scale = exp2(float(exp));
	
	vec3 v;
	v.x = float(bitfieldExtract(x[irr_glsl_RGB19E7_COMPONENT_INDICES[0]], irr_glsl_RGB19E7_COMPONENT_BITOFFSETS[0], irr_glsl_RGB19E7_MANTISSA_BITS));
	v.y = float(bitfieldInsert(
		bitfieldExtract(x[irr_glsl_RGB19E7_COMPONENT_INDICES[1]], irr_glsl_RGB19E7_COMPONENT_BITOFFSETS[1], irr_glsl_RGB19E7_G_COMPONENT_SPLIT),
		bitfieldExtract(x[irr_glsl_RGB19E7_COMPONENT_INDICES[2]], 0, irr_glsl_RGB19E7_COMPONENT_BITOFFSETS[2]),
		irr_glsl_RGB19E7_G_COMPONENT_SPLIT,
		irr_glsl_RGB19E7_COMPONENT_BITOFFSETS[2]
	));
	v.z = float(bitfieldExtract(x[irr_glsl_RGB19E7_COMPONENT_INDICES[2]], irr_glsl_RGB19E7_COMPONENT_BITOFFSETS[2], irr_glsl_RGB19E7_MANTISSA_BITS));
	
	return v*scale;
}

vec4 irr_glsl_decodeRGB10A2(in uint x)
{
	uvec4 shifted = uvec4(x,uvec3(x)>>uvec3(10,20,30));
	const uvec3 rgbMask = uvec3(0x3ffu);
	return vec4(vec3(shifted.rgb&rgbMask),shifted.a)/vec4(vec3(rgbMask),3.0);
}

#endif