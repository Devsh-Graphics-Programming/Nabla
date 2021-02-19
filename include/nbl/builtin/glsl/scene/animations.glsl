#ifndef _NBL_GLSL_SCENE_ANIMATIONS_INCLUDED_
#define _NBL_GLSL_SCENE_ANIMATIONS_INCLUDED_


#include <nbl/builtin/glsl/math/keyframe.glsl>


struct nbl_glsl_scene_Animation_t
{
	uint keyframeOffset; // same offset for timestamps and keyframes
	uint keyframesCount_interpolationMode; // 2 bits for interpolation mode
};

struct nbl_glsl_scene_AnimationBlend_t
{
	uint nodeTargetID;
	uint desiredTimestamp;
	float weight;
	uint animationID; // or do we stick whole animation inside?
};

/*

Design Idea

IAnimationLibrary
{

Packs several ranges of nbl_glsl_scene_Keyframe_t and uint-ms-timestamps into a single buffer (timestamps and keyframes go into separate buffers)

The packed ranges are named and stored as `map<const char*,nbl_glsl_scene_Animation_t*>`

You can retrieve animations by name or ID

}

*/

#endif