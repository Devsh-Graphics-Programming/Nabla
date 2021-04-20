#ifndef _NBL_GLSL_SCENE_KEYFRAME_INCLUDED_
#define _NBL_GLSL_SCENE_KEYFRAME_INCLUDED_



#include <nbl/builtin/glsl/math/quaternions.glsl>



struct nbl_glsl_scene_Keyframe_t
{
	uvec2 data[3];
};


vec3 nbl_glsl_scene_Keyframe_t_getScale(in nbl_glsl_scene_Keyframe_t keyframe)
{
	return nbl_glsl_decodeRGB18E7S3(keyframe.data[2]);
}

nbl_glsl_quaternion_t nbl_glsl_scene_Keyframe_t_getRotation(in nbl_glsl_scene_Keyframe_t keyframe)
{
	return {nbl_glsl_decode8888Quaternion(keyframe.data[1][1])};
}

vec3 nbl_glsl_scene_Keyframe_t_getTranslation(in nbl_glsl_scene_Keyframe_t keyframe)
{
	return uintBitsToFloat(uvec3(keyframe.data[0].xy, keyframe.data[1][0]));
}



struct nbl_glsl_scene_FatKeyframe_t
{
	vec3 scale;
	nbl_glsl_quaternion_t rotation;
	vec3 translation;
};


nbl_glsl_scene_FatKeyframe_t nbl_glsl_scene_FatKeyframe_t_FatKeyframe_t(in nbl_glsl_scene_Keyframe_t keyframe)
{
	nbl_glsl_scene_FatKeyframe_t result;
	result.scale = nbl_glsl_scene_Keyframe_t_getScale(keyframe);
	result.rotation = nbl_glsl_scene_Keyframe_t_getRotation(keyframe);
	result.translation = nbl_glsl_scene_Keyframe_t_getTranslation(keyframe);
	return result;
}

nbl_glsl_scene_FatKeyframe_t nbl_glsl_scene_FatKeyframe_t_interpolate(in nbl_glsl_scene_FatKeyframe_t start, in nbl_glsl_scene_FatKeyframe_t end, in float fraction)
{
	nbl_glsl_scene_FatKeyframe_t result;
	result.scale = mix(start.scale,end.scale,fraction);
	result.rotation = nbl_glsl_quaternion_t_flerp(start.rotation,end.rotation,fraction);
	result.translation = mix(start.translation,end.translation,fraction);
	return result;
}

mat4x3 nbl_glsl_scene_FatKeyframe_t_constructMatrix(in nbl_glsl_scene_FatKeyframe_t keyframe)
{
	mat3 rotation = nbl_glsl_quaternion_t_constructMatrix(keyframe.rotation);
	mat4x3 tform = mat4x3(rotation[0],rotation[1],rotation[2],keyframe.translation);
	for (int i=0; i<3; i++)
		tform[i] *= keyframe.scale[i];
	return tform;
}

#endif