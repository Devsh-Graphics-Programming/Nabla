#version 460 core

layout(location = 0) in uint iNodeID; // instance data
layout(location = 1) in float iScale; // instance data

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

layout(location = 0) out float outIScale;
layout(location = 1) out mat4x3 outNodeGlobalTransform;

#define INVALID 3735928559u
#include "nbl/builtin/glsl/utils/transform.glsl"

void main()
{
    const uint parent = nodeParents.data[iNodeID];
    vec3 lineWorldPos = nodeGlobalTransforms.data[bool(gl_VertexIndex&0x1u)&&parent!=INVALID ? parent:iNodeID][3];

    gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,lineWorldPos);
	
	outNodeGlobalTransform = nodeGlobalTransforms.data[iNodeID];
	outIScale = iScale;
}