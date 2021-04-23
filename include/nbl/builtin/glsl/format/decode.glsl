#ifndef _NBL_BUILTIN_GLSL_FORMAT_DECODE_INCLUDED_
#define _NBL_BUILTIN_GLSL_FORMAT_DECODE_INCLUDED_

#include <nbl/builtin/glsl/format/constants.glsl>
#include <nbl/builtin/glsl/math/quaternions.glsl>

vec3 nbl_glsl_decodeRGB19E7(in uvec2 x)
{
	int exp = int(bitfieldExtract(x[nbl_glsl_RGB19E7_COMPONENT_INDICES[3]], nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[3], nbl_glsl_RGB19E7_EXPONENT_BITS) - nbl_glsl_RGB19E7_EXP_BIAS - nbl_glsl_RGB19E7_MANTISSA_BITS);
	float scale = exp2(float(exp));
	
	vec3 v;
	v.x = float(bitfieldExtract(x[nbl_glsl_RGB19E7_COMPONENT_INDICES[0]], nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[0], nbl_glsl_RGB19E7_MANTISSA_BITS));
	v.y = float(bitfieldInsert(
		bitfieldExtract(x[nbl_glsl_RGB19E7_COMPONENT_INDICES[1]], nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[1], nbl_glsl_RGB19E7_G_COMPONENT_SPLIT),
		bitfieldExtract(x[nbl_glsl_RGB19E7_COMPONENT_INDICES[2]], 0, nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[2]),
		nbl_glsl_RGB19E7_G_COMPONENT_SPLIT,
		nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[2]
	));
	v.z = float(bitfieldExtract(x[nbl_glsl_RGB19E7_COMPONENT_INDICES[2]], nbl_glsl_RGB19E7_COMPONENT_BITOFFSETS[2], nbl_glsl_RGB19E7_MANTISSA_BITS));
	
	return v*scale;
}

vec3 nbl_glsl_decodeRGB18E7S3(in uvec2 x)
{
	int exp = int(bitfieldExtract(x[nbl_glsl_RGB18E7S3_COMPONENT_INDICES[3]], nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS[3], nbl_glsl_RGB18E7S3_EXPONENT_BITS) - nbl_glsl_RGB18E7S3_EXP_BIAS - nbl_glsl_RGB18E7S3_MANTISSA_BITS);
	float scale = exp2(float(exp));
	
	vec3 v;
	v.x = float(bitfieldExtract(x[nbl_glsl_RGB18E7S3_COMPONENT_INDICES[0]], nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS[0], nbl_glsl_RGB18E7S3_MANTISSA_BITS));
	v.y = float(bitfieldInsert(
		bitfieldExtract(x[nbl_glsl_RGB18E7S3_COMPONENT_INDICES[1]], nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS[1], nbl_glsl_RGB18E7S3_G_COMPONENT_SPLIT),
		bitfieldExtract(x[nbl_glsl_RGB18E7S3_COMPONENT_INDICES[2]], 0, nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS[2]),
		nbl_glsl_RGB18E7S3_G_COMPONENT_SPLIT,
		nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS[2]
	));
	v.z = float(bitfieldExtract(x[nbl_glsl_RGB18E7S3_COMPONENT_INDICES[2]], nbl_glsl_RGB18E7S3_COMPONENT_BITOFFSETS[2], nbl_glsl_RGB18E7S3_MANTISSA_BITS));
	
	uvec3 signs = x.yyy<<uvec3(2u,1u,0u);
	signs &= 0x80000000u;
	v = uintBitsToFloat(floatBitsToUint(v)^signs);

	return v*scale;
}

vec4 nbl_glsl_decodeRGB10A2_UNORM(in uint x)
{
	const uvec3 rgbMask = uvec3(0x3ffu);
	const uvec4 shifted = uvec4(x,uvec3(x)>>uvec3(10,20,30));
	return vec4(vec3(shifted.rgb&rgbMask),shifted.a)/vec4(vec3(rgbMask),3.0);
}
vec4 nbl_glsl_decodeRGB10A2_SNORM(in uint x)
{
	const ivec4 shifted = ivec4(x, uvec3(x) >> uvec3(10u, 20u, 30u));
	const ivec4 rgbaBias = ivec4(ivec3(0x200u), 0x2u);
	const ivec4 halfMask = rgbaBias - ivec4(1);
	const ivec4 signed = (-(shifted & rgbaBias)) | (shifted & halfMask);
	return max(vec4(signed) / vec4(halfMask), vec4(-1.0));
}


nbl_glsl_quaternion_t nbl_glsl_decode8888Quaternion(in uint x)
{
	nbl_glsl_quaternion_t quat;
	quat.data = normalize(unpackSnorm4x8(x));
	return quat;
}

nbl_glsl_quaternion_t nbl_glsl_decode1010102Quaternion(in uint x)
{
	const uvec3 rgbMask = uvec3(0x3ffu);
	const uvec4 shifted = uvec4(x,uvec3(x)>>uvec3(10,20,30));
	const uint maxCompIx = shifted[3];

	const ivec3 maxVal = ivec3(0x1ff);
	const ivec3 unnorm = max(-maxVal,ivec3(shifted.rgb&rgbMask)-ivec3(maxVal+1));
	const vec3 smallest3Components = vec3(unnorm)*inversesqrt(2.f)/vec3(maxVal);

	nbl_glsl_quaternion_t quat;
	quat.data[maxCompIx>0u ? 0:1] = smallest3Components[0];
	quat.data[maxCompIx>1u ? 1:2] = smallest3Components[1];
	quat.data[maxCompIx>2u ? 2:3] = smallest3Components[2];
	quat.data[maxCompIx] = sqrt(1.0-dot(smallest3Components,smallest3Components));
	return quat;
}

#endif