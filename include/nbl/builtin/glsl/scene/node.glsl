#ifndef _NBL_GLSL_SCENE_NODE_INCLUDED_
#define _NBL_GLSL_SCENE_NODE_INCLUDED_


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


struct nbl_glsl_scene_Node_dynamic_data_t
{
	mat4x3	relativeTransformation;
	mat4x3	globalTransformation;
	float	animationTime;
	uint	padding[3];
};

struct nbl_glsl_scene_Node_per_camera_data_t
{
	mat4 worldViewProj;
};


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

#endif