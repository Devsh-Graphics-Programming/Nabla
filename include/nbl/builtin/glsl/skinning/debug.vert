#version 460 core

#include "nbl/builtin/glsl/skinning/cache_descriptor_set.glsl"

#include "nbl/builtin/glsl/shapes/aabb.glsl"
#ifndef NBL_GLSL_SKINNING_DEBUG_DESCRIPTOR_SET
#define NBL_GLSL_SKINNING_DEBUG_DESCRIPTOR_SET 1
#endif
#ifndef NBL_GLSL_SKINNING_DEBUG_AABB_DESCRIPTOR_BINDING
#define NBL_GLSL_SKINNING_DEBUG_AABB_DESCRIPTOR_BINDING 0
#endif
#ifndef NBL_GLSL_SKINNING_DEBUG_AABB_DESCRIPTOR_QUALIFIERS
#define NBL_GLSL_SKINNING_DEBUG_AABB_DESCRIPTOR_QUALIFIERS readonly restrict
#endif
#ifndef NBL_GLSL_SKINNING_DEBUG_AABB_DESCRIPTOR_DECLARED
#define NBL_GLSL_SKINNING_DEBUG_AABB_DESCRIPTOR_DECLARED
layout(
    set=NBL_GLSL_SKINNING_DEBUG_DESCRIPTOR_SET,
    binding=NBL_GLSL_SKINNING_DEBUG_AABB_DESCRIPTOR_BINDING
) NBL_GLSL_SKINNING_DEBUG_AABB_DESCRIPTOR_QUALIFIERS buffer DebugAABB
{
    nbl_glsl_shapes_CompressedAABB_t data[];
} debugAABB;
#endif


layout( push_constant, row_major ) uniform PushConstants
{
    mat4 viewProj;
	vec4 aabbColor;
} pc;


layout(location = 0) out vec3 outColor;


#include "nbl/builtin/glsl/utils/transform.glsl"
void main()
{
	// TODO: upper_bound based off gl_InstanceIndex, map to jointID + skinInstanceOffset
/*
	const nbl_glsl_shapes_AABB_t aabb = nbl_glsl_shapes_CompressedAABB_t_decompress(debugAABB.data[aabbID]);

	const bvec3 mask = bvec3(gl_VertexIndex&0x1u,gl_VertexIndex&0x2u,gl_VertexIndex&0x4u);
	const vec3 pos = nbl_glsl_pseudoMul3x4with3x1(nodeGlobalTransforms.data[pivotNodeID]*skinningTransforms[jointID],mix(aabb.minVx,aabb.maxVx,mask));
*/
	outColor = pc.aabbColor.rgb;
	gl_Position = vec4(6.f,0.f,0.f,9.f);//nbl_glsl_pseudoMul4x4with3x1(pc.viewProj,pos);

}