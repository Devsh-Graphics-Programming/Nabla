#ifndef _NBL_GLSL_SKINNING_LINEAR_GLSL_INCLUDED_
#define _NBL_GLSL_SKINNING_LINEAR_GLSL_INCLUDED_

#include <nbl/builtin/glsl/utils/transform.glsl>
void nbl_glsl_skinning_linear(inout vec3 modelspacePos, inout vec3 modelspaceNormal, in mat4x3 blendedTform)
{
	modelspacePos = nbl_glsl_pseudoMul3x4with3x1(blendedTform,modelspacePos);

	mat3 sub3x3TransposeCofactors;
	const uint signFlip = nbl_glsl_sub3x3TransposeCofactors(mat3(blendedTform), sub3x3TransposeCofactors);
	modelspaceNormal = nbl_glsl_fastNormalTransform(signFlip,sub3x3TransposeCofactors,modelspaceNormal);
}


#include <nbl/builtin/glsl/skinning/render_descriptor_set.glsl>
mat4x3 nbl_glsl_skinning_getJointMatrix(in uint skinCacheOffset, in uint jointID);

void nbl_glsl_skinning_linear(inout vec3 modelspacePos, inout vec3 modelspaceNormal, in uint skinCacheOffset, in uint jointID)
{
	nbl_glsl_skinning_linear(modelspacePos,modelspaceNormal,nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID));
}
void nbl_glsl_skinning_linear(inout vec3 modelspacePos, inout vec3 modelspaceNormal, in uint skinCacheOffset, in uvec2 jointID, in float jointWeight)
{
	mat4x3 skinTform = nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID[0])*jointWeight;
	if (jointWeight!=1.f)
		skinTform += nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID[1])*(1.f-jointWeight);

	nbl_glsl_skinning_linear(modelspacePos,modelspaceNormal,skinTform);
}
void nbl_glsl_skinning_linear(inout vec3 modelspacePos, inout vec3 modelspaceNormal, in uint skinCacheOffset, in uvec3 jointID, in vec2 jointWeight)
{
	mat4x3 skinTform = nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID[0])*jointWeight[0];
	float lastWeight = 1.f-jointWeight[0];
	if (jointWeight[1]!=0.f)
	{
		skinTform += nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID[1])*jointWeight[1];
		lastWeight -= jointWeight[1];
	}
	if (lastWeight!=0.f)
		skinTform += nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID[2])*lastWeight;

	nbl_glsl_skinning_linear(modelspacePos,modelspaceNormal,skinTform);
}
void nbl_glsl_skinning_linear(inout vec3 modelspacePos, inout vec3 modelspaceNormal, in uint skinCacheOffset, in uvec4 jointID, in vec3 jointWeight)
{
	mat4x3 skinTform = nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID[0])*jointWeight[0];
	float lastWeight = 1.f-jointWeight[0];
	if (jointWeight[1]!=0.f)
	{
		skinTform += nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID[1])*jointWeight[1];
		lastWeight -= jointWeight[1];
	}
	if (jointWeight[2]!=0.f)
	{
		skinTform += nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID[2])*jointWeight[2];
		lastWeight -= jointWeight[2];
	}
	if (lastWeight!=0.f)
		skinTform += nbl_glsl_skinning_getJointMatrix(skinCacheOffset,jointID[3])*lastWeight;

	nbl_glsl_skinning_linear(modelspacePos,modelspaceNormal,skinTform);
}

#include <nbl/builtin/glsl/skinning/render_descriptor_set.glsl>
mat4x3 nbl_glsl_skinning_getJointMatrix(in uint skinCacheOffset, in uint jointID)
{
	return skinningTransforms.data[skinCacheOffset+jointID];
}

#endif