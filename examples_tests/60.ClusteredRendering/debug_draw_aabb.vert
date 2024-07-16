#version 460 core

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>
#include <nbl/builtin/glsl/shapes/aabb.glsl>

layout (set = 0, binding = 0) restrict buffer readonly DebugAABBs
{
	nbl_glsl_shapes_AABB_t data[];
} debugAABBs;

layout (set = 0, binding = 1, row_major, std140) uniform UBO
{
    nbl_glsl_SBasicViewParameters params;
} CamData;

layout(location = 0) out vec3 color;

void main()
{
	const nbl_glsl_shapes_AABB_t aabb = debugAABBs.data[gl_InstanceIndex];

	const bvec3 mask = bvec3(gl_VertexIndex&0x1u,gl_VertexIndex&0x2u,gl_VertexIndex&0x4u);
	vec3 pos = mix(aabb.minVx,aabb.maxVx,mask);

	gl_Position = nbl_glsl_pseudoMul4x4with3x1(CamData.params.MVP, pos);
	color = vec3(1.f, 0.f, 0.f);
}