#version 460 core
layout(location = 14) in uint nodeID;
layout(location = 15) in uint aabbID;


#include "nbl/builtin/glsl/transform_tree/pool_descriptor_set.glsl"


#include "nbl/builtin/glsl/shapes/aabb.glsl"

#ifndef NBL_GLSL_TRANSFORM_TREE_DEBUG_DESCRIPTOR_SET
#define NBL_GLSL_TRANSFORM_TREE_DEBUG_DESCRIPTOR_SET 1
#endif
#ifndef NBL_GLSL_TRANSFORM_TREE_DEBUG_AABB_DESCRIPTOR_BINDING
#define NBL_GLSL_TRANSFORM_TREE_DEBUG_AABB_DESCRIPTOR_BINDING 0
#endif
#ifndef NBL_GLSL_TRANSFORM_TREE_DEBUG_AABB_DESCRIPTOR_QUALIFIERS
#define NBL_GLSL_TRANSFORM_TREE_DEBUG_AABB_DESCRIPTOR_QUALIFIERS readonly restrict
#endif
#ifndef NBL_GLSL_TRANSFORM_TREE_DEBUG_AABB_DESCRIPTOR_DECLARED
#define NBL_GLSL_TRANSFORM_TREE_DEBUG_AABB_DESCRIPTOR_DECLARED
layout(
    set=NBL_GLSL_TRANSFORM_TREE_DEBUG_DESCRIPTOR_SET,
    binding=NBL_GLSL_TRANSFORM_TREE_DEBUG_AABB_DESCRIPTOR_BINDING
) NBL_GLSL_TRANSFORM_TREE_DEBUG_AABB_DESCRIPTOR_QUALIFIERS buffer DebugAABB
{
    nbl_glsl_shapes_CompressedAABB_t data[];
} debugAABB;
#endif


layout( push_constant, row_major ) uniform PushConstants
{
    mat4 viewProj;
	vec4 lineColor;
	vec4 aabbColor;
} pc;


layout(location = 0) out vec3 outColor;


#include "nbl/builtin/glsl/utils/transform.glsl"
void main()
{
	vec3 pos;
	if(gl_VertexIndex<8u) // render box
	{
		const nbl_glsl_shapes_AABB_t aabb = nbl_glsl_shapes_CompressedAABB_t_decompress(debugAABB.data[aabbID]);

		const bvec3 mask = bvec3(gl_VertexIndex&0x1u,gl_VertexIndex&0x2u,gl_VertexIndex&0x4u);
		pos = mix(aabb.minVx,aabb.maxVx,mask);

		outColor = pc.aabbColor.rgb;
	}
	else // render node-parent line
	{
		uint id = nodeID;
		uint nodeParentID = nodeParents.data[nodeID];
		
		if (bool(gl_VertexIndex&0x1u)&&nodeParentID!=NBL_GLSL_PROPERTY_POOL_INVALID)
			id = nodeParentID;
		pos = nodeGlobalTransforms.data[id][3];

		outColor = pc.lineColor.rgb;
	}
	gl_Position = nbl_glsl_pseudoMul4x4with3x1(pc.viewProj,pos);
}