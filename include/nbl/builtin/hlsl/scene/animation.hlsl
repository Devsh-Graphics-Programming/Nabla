
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SCENE_ANIMATION_INCLUDED_
#define _NBL_BUILTIN_HLSL_SCENE_ANIMATION_INCLUDED_

#include <nbl/builtin/hlsl/scene/keyframe.hlsl>

namespace nbl
{
namespace hlsl
{
namespace scene
{

struct Animation_t
{
	uint keyframeOffset; // same offset for timestamps and keyframes
	
	uint keyframesCount : 30;       // 30 bits for keyframes count
	uint interpolationMode : 2;     // 2 bits for interpolation mode
};


// TODO: Design for this
struct AnimationBlend_t
{
	uint nodeTargetID;
	uint desiredTimestamp;
	float weight;
	uint animationID; // or do we stick whole animation inside?
};

}
}
}

#endif