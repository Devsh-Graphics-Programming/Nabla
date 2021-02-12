#ifndef _NBL_GLSL_SCENE_ANIMATIONS_INCLUDED_
#define _NBL_GLSL_SCENE_ANIMATIONS_INCLUDED_



#include <nbl/builtin/glsl/math/keyframe.glsl>



struct nbl_glsl_scene_Animations_t
{
	mat4x3 poseBind;
	uint keyframesCount;
	uint timestampsOffset;
	// The difference between the two is because some animation formats allow for staggered keyframes
	// i.e. not every keyframe is a tuple of (scale,rotation,translation)
	uint interpolatedKeyframesOffset;
	uint nonInterpolatedKeyframesOffset;
};



#endif