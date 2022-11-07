
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

		float3 getScale();
		quaternion_t getRotation();
		float3 getTranslation();
	};



	struct FatKeyframe_t
	{
		float3 scale;
		quaternion_t rotation;
		float3 translation;

		FatKeyframe_t FatKeyframe_t(in Keyframe_t keyframe);
		FatKeyframe_t interpolate(in FatKeyframe_t start, in FatKeyframe_t end, in float fraction);
		Float4x3 constructMatrix()
	};
} 
}
}





#endif