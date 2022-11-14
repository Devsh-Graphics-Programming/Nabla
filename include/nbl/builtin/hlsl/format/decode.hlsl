
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_FORMAT_DECODE_INCLUDED_
#define _NBL_BUILTIN_HLSL_FORMAT_DECODE_INCLUDED_

#include <nbl/builtin/hlsl/common.hlsl>
#include <nbl/builtin/hlsl/format/constants.hlsl>
#include <nbl/builtin/hlsl/math/quaternions.hlsl>

namespace nbl
{
namespace hlsl
{
namespace format
{


float3 decodeRGB19E7(in uint2 x)
{
	int exp = int(bitfieldExtract(x[RGB19E7_COMPONENT_INDICES[3]], RGB19E7_COMPONENT_BITOFFSETS[3], RGB19E7_EXPONENT_BITS) - RGB19E7_EXP_BIAS - RGB19E7_MANTISSA_BITS);
	float scale = exp2(float(exp));
	
	float3 v;
	v.x = float(bitfieldExtract(x[RGB19E7_COMPONENT_INDICES[0]], RGB19E7_COMPONENT_BITOFFSETS[0], RGB19E7_MANTISSA_BITS));
	v.y = float(bitfieldInsert(
		bitfieldExtract(x[RGB19E7_COMPONENT_INDICES[1]], RGB19E7_COMPONENT_BITOFFSETS[1], RGB19E7_G_COMPONENT_SPLIT),
		bitfieldExtract(x[RGB19E7_COMPONENT_INDICES[2]], 0, RGB19E7_COMPONENT_BITOFFSETS[2]),
		RGB19E7_G_COMPONENT_SPLIT,
		RGB19E7_COMPONENT_BITOFFSETS[2]
	));
	v.z = float(bitfieldExtract(x[RGB19E7_COMPONENT_INDICES[2]], RGB19E7_COMPONENT_BITOFFSETS[2], RGB19E7_MANTISSA_BITS));
	
	return v*scale;
}

float3 decodeRGB18E7S3(in uint2 x)
{
	int exp = int(bitfieldExtract(x[RGB18E7S3_COMPONENT_INDICES[3]], RGB18E7S3_COMPONENT_BITOFFSETS[3], RGB18E7S3_EXPONENT_BITS) - RGB18E7S3_EXP_BIAS - RGB18E7S3_MANTISSA_BITS);
	float scale = exp2(float(exp));
	
	float3 v;
	v.x = float(bitfieldExtract(x[RGB18E7S3_COMPONENT_INDICES[0]], RGB18E7S3_COMPONENT_BITOFFSETS[0], RGB18E7S3_MANTISSA_BITS));
	v.y = float(bitfieldInsert(
		bitfieldExtract(x[RGB18E7S3_COMPONENT_INDICES[1]], RGB18E7S3_COMPONENT_BITOFFSETS[1], RGB18E7S3_G_COMPONENT_SPLIT),
		bitfieldExtract(x[RGB18E7S3_COMPONENT_INDICES[2]], 0, RGB18E7S3_COMPONENT_BITOFFSETS[2]),
		RGB18E7S3_G_COMPONENT_SPLIT,
		RGB18E7S3_COMPONENT_BITOFFSETS[2]
	));
	v.z = float(bitfieldExtract(x[RGB18E7S3_COMPONENT_INDICES[2]], RGB18E7S3_COMPONENT_BITOFFSETS[2], RGB18E7S3_MANTISSA_BITS));
	
	uint3 signs = x.yyy<<uint3(2u,1u,0u);
	signs &= 0x80000000u;
	v = asfloat(asuint(v)^signs);

	return v*scale;
}

//
float4 decodeRGB10A2_UNORM(in uint x)
{
	const uint3 rgbMask = (0x3ffu).xxx;
	const uint4 shifted = uint4(x, (x).xxx >> uint3(10,20,30));
	return float4(float3(shifted.rgb&rgbMask),shifted.a)/float4(float3(rgbMask),3.0);
}
float4 decodeRGB10A2_SNORM(in uint x)
{
	const int4 shifted = int4(x, (x).xxx >> uint3(10u,20u,30u));
	const int4 rgbaBias = int4((0x200u).xxx, 0x2u);
	const int4 halfMask = rgbaBias - (1).xxxx;
	const int4 _signed = (-(shifted & rgbaBias)) | (shifted & halfMask);
	return max(float4(_signed) / float4(halfMask), (-1.f).xxxx);
}

//
quaternion_t decode8888Quaternion(in uint x)
{
	quaternion_t quat;
	quat.data = normalize(unpackSnorm4x8(x));
	return quat;
}

quaternion_t decode1010102Quaternion(in uint x)
{
	const uint3 rgbMask = (0x3ffu).xxx;
	const uint4 shifted = uint4(x, (x).xxx >> uint3(10,20,30));
	const uint maxCompIx = shifted[3];

	const int3 maxVal = (0x1ff).xxx;
	const int3 unnorm = max(-maxVal,int3(shifted.rgb&rgbMask)-int3(maxVal+1));
	const float3 smallest3Components = float3(unnorm)*rsqrt(2.f)/float3(maxVal);

	quaternion_t quat;
	quat.data[select(maxCompIx > 0u, 0, 1)] = smallest3Components[0];
	quat.data[select(maxCompIx > 1u, 1, 2)] = smallest3Components[1];
	quat.data[select(maxCompIx > 2u, 2, 3)] = smallest3Components[2];
	quat.data[maxCompIx] = sqrt(1.0-dot(smallest3Components,smallest3Components));
	return quat;
}


}
}
}

#endif