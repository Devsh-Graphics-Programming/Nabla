#version 460 core

layout(location = 0) in uint iNodeID; // instance data

layout(binding = 0, std430) restrict readonly buffer NodeParents
{
    uint data[];
} nodeParents;

layout(binding = 3, std430) restrict readonly buffer NodeGlobalTransforms
{
    layout(row_major) mat4x3 data[];
} nodeGlobalTransforms;

layout( push_constant, row_major ) uniform Block
{
    mat4 viewProj;
	vec4 color;
} PushConstants;

layout(location = 0) out vec4 outColor;

#define INVALID 3735928559u
#include "nbl/builtin/glsl/utils/transform.glsl"

void main()
{
    const uint parent = nodeParents.data[iNodeID];
    vec3 pos = nodeGlobalTransforms.data[bool(gl_VertexIndex&0x1u)&&parent!=INVALID ? parent:iNodeID][3];

    gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,pos);
	outColor = PushConstants.color;
}