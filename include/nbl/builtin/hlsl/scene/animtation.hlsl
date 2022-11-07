
#ifndef _NBL_HLSL_SCENE_ANIMATION_INCLUDED_
#define _NBL_HLSL_SCENE_ANIMATION_INCLUDED_


#include <nbl/builtin/hlsl/math/keyframe.hlsl>

namespace nbl
{
	namespace hlsl
	{
		namespace scene
		{
			struct Animation_t
			{
				uint keyframeOffset; // same offset for timestamps and keyframes
				uint keyframesCount_interpolationMode; // 2 bits for interpolation mode
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