
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_SCENE_KEYFRAME_INCLUDED_
#define _NBL_HLSL_SCENE_KEYFRAME_INCLUDED_



#include <nbl/builtin/hlsl/math/quaternions.hlsl>



namespace nbl
{
	namespace hlsl
	{
		namespace scene
		{
			struct Keyframe_t
			{
				uint2 data[3];

				float3 getScale(in Keyframe_t keyframe);
				quaternion_t getRotation(in Keyframe_t keyframe);
				float3 getTranslation(in Keyframe_t keyframe);
			};




			struct FatKeyframe_t
			{
				float3 scale;
				quaternion_t rotation;
				float3 translation;

				FatKeyframe_t FatKeyframe_t(in Keyframe_t keyframe);
			};
		} 
	}
}





#endif
