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
	vec4 lineColor;
	vec4 aabbColor;
	vec4 minEdge;
	vec4 maxEdge;
} PushConstants;

layout(location = 0) out vec3 outGlobalTMinEdge;
layout(location = 1) out vec3 outGlobalTMaxEdge;

#define INVALID 3735928559u
#include "nbl/builtin/glsl/utils/transform.glsl"

void main()
{
    const uint parent = nodeParents.data[iNodeID];
    vec3 lineWorldPos = nodeGlobalTransforms.data[bool(gl_VertexIndex&0x1u)&&parent!=INVALID ? parent:iNodeID][3];

    gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,lineWorldPos);
	
	mat4x3 nodeGlobalTransform = nodeGlobalTransforms.data[iNodeID];
	
	outGlobalTMinEdge = nbl_glsl_pseudoMul3x4with3x1(nodeGlobalTransform,PushConstants.minEdge.xyz);
	outGlobalTMaxEdge = nbl_glsl_pseudoMul3x4with3x1(nodeGlobalTransform,PushConstants.maxEdge.xyz);
}