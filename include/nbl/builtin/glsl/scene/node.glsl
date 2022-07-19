#ifndef _NBL_GLSL_SCENE_NODE_INCLUDED_
#define _NBL_GLSL_SCENE_NODE_INCLUDED_



#include <nbl/builtin/glsl/math/animations.glsl>

void nbl_glsl_scene_Node_initializeLinearSkin(out vec4 accVertexPos, out vec3 accVertexNormal, in vec3 inVertexPos, in vec3 inVertexNormal, in mat4 boneTransform, in mat3 boneOrientationInvT, in float boneWeight)
{
	accVertexPos = boneTransform * vec4(inVertexPos * boneWeight, boneWeight);
	accVertexNormal = boneOrientationInvT * inVertexNormal * boneWeight;
}
void nbl_glsl_scene_Node_accumulateLinearSkin(inout vec4 accVertexPos, inout vec3 accVertexNormal, in vec3 inVertexPos, in vec3 inVertexNormal, in mat4 boneTransform, in mat3 boneOrientationInvT, in float boneWeight)
{
	accVertexPos += boneTransform * vec4(inVertexPos * boneWeight, boneWeight);
	accVertexNormal += boneOrientationInvT * inVertexNormal * boneWeight;
}

#endif