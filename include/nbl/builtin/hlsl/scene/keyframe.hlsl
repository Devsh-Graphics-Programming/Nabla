
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SCENE_KEYFRAME_INCLUDED_
#define _NBL_BUILTIN_HLSL_SCENE_KEYFRAME_INCLUDED_

#include <nbl/builtin/hlsl/math/quaternions.hlsl>

namespace nbl
{
namespace hlsl
{
namespace scene
{

using namespace math;

struct Keyframe_t
{
	uint2 data[3];

	float3 getScale()
	{
		return decodeRGB18E7S3(data[2]);
	}

	quaternion_t getRotation()
	{
		return { decode8888Quaternion(data[1][1]) };
	}

	float3 getTranslation()
	{
		return asfloat(uint3(data[0].xy, data[1][0]));
	}
};


struct FatKeyframe_t
{
	float3 scale;
	quaternion_t rotation;
	float3 translation;

	FatKeyframe_t decompress(const Keyframe_t keyframe)
	{
		FatKeyframe_t result;

		result.scale       = keyframe.getScale();
		result.rotation    = keyframe.getRotation();
		result.translation = keyframe.getTranslation();

		return result;
	}

	FatKeyframe_t interpolate(const FatKeyframe_t start, const FatKeyframe_t end, const float fraction)
	{
		FatKeyframe_t result;

		result.scale = lerp(start.scale, end.scale, fraction);
		result.rotation = quaternion_t::flerp(start.rotation, end.rotation, fraction);
		result.translation = lerp(start.translation, end.translation, fraction);

		return result;
	}

	float3x4 constructMatrix(const FatKeyframe_t keyframe)
	{
		float3x3 rotation = keyframe.rotation.constructMatrix(keyframe.rotation);
		float3x4 tform; = float3x4(rotation[0], rotation[1], rotation[2], keyframe.translation);

		for (int i=0; i<3; i++)
			tform[i] = float4(rotation[i]*scale,keyframe.translation[i]);

		return tform;
	}
};

}
}
}

#endif