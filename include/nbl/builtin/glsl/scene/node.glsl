#ifndef _NBL_GLSL_SCENE_NODE_INCLUDED_
#define _NBL_GLSL_SCENE_NODE_INCLUDED_



const uint nbl_glsl_scene_Node_invalid_UID = 0xdeadbeefu;

bool nbl_glsl_scene_Node_isValidUID(in uint uid)
{
	return uid!=nbl_glsl_scene_Node_invalid_UID;
}
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


struct nbl_glsl_scene_Node_static_data_t
{
	vec3 AABBMin;
	uint parentUID;
	vec3 AABBMax;
	uint flags;
};
const uint nbl_glsl_scene_Node_static_data_t_EF_CULL_CHILDREN = 0x00000002u;
const uint nbl_glsl_scene_Node_static_data_t_EF_DRAWABLE = 0x00000001u;

mat2x3 nbl_glsl_scene_Node_static_data_t_getAABB(in nbl_glsl_scene_Node_static_data_t node)
{
	return mat2x3(node.AABBMin,node.AABBMax);
}


struct nbl_glsl_scene_Node_animation_data_t
{
	mat4x3	relativeTransformation;
	float	animationTime;
};


struct nbl_glsl_scene_Node_output_data_t
{
	mat4x3	globalTransformation;
	vec3	globalNormalMatrixRow0;
	uint	padding0;
	vec3	globalNormalMatrixRow1;
	uint	padding1;
	vec3	globalNormalMatrixRow2;
	uint	padding2;
};

mat3 nbl_glsl_scene_Node_output_data_t_getNormalMatrix(in nbl_glsl_scene_Node_output_data_t node_output_data)
{
	return mat3(node_output_data.globalNormalMatrixRow0,node_output_data.globalNormalMatrixRow1,node_output_data.globalNormalMatrixRow2);
}


struct nbl_glsl_scene_Node_per_camera_data_t
{
	mat4 worldViewProj;
};

/*
bool nodeIsVisible(in nbl_glsl_scene_Node_static_data_t node_s, in nbl_glsl_scene_Node_dynamic_data_t node_d)
{
	bool visible = false;
	for (uint i=0u; i<pc.CameraCount; i++)
	{
		mat4 worldViewProj = cameras[i].ViewProj*node_d.globalTransformation;
		if (!nbl_couldBeVisible(worldViewProj,nbl_glsl_scene_Node_static_data_t_getAABB(node_s)))
			continue;
		visible = true;
		per_camera_data[cameras[i].nodeDataOffset+stuff].worldViewProj = worldViewProj;
	}
	return visible;
}

bool checkUpdateStamp(in uint newStamp, in uint nodeUID)
{
	memoryBarrierBuffer();
	return newStamp!=NodeUpdateStamp[nodeUID];
}

void update(in uint newStamp, in uint nodeUID)
{
	const nbl_glsl_scene_Node_static_data_t node_s = StaticNodeData[nodeUID];
	// see what is ready
	uint ancestorStackSize = 0u;
	uint ancestorStack[kMaxHierarchyDepth-1u]; // `kMaxHierarchyDepth` must be at least 2
	for (uint parentUID=node_s.parentUID; checkUpdateStamp(newStamp,parentUID); parentUID=StaticNodeData[parentUID].parentUID)
		ancestorStack[ancestorStackSize++] = parentUID;
	// double compute what is not ready
	while (ancestorStackSize--)
	{
		//
	}

}
*/

#endif