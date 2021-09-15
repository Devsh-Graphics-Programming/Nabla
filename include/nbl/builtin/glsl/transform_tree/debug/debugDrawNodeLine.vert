#version 460 core

layout(location = 15) in uint iNodeID;

layout(set=0, binding=0, row_major) readonly buffer NodeTransforms
{
    uint data[];
} nodeParent;
layout(set=0, binding=1, row_major) readonly buffer NodeTransforms
{
    mat4x3 data[];
} nodeGlobalTransforms;

layout( push_constant, row_major ) uniform Block
{
    mat4 viewProj;
	vec4 color;
} PushConstants;

layout(location = 0) out vec4 outColor;

#include "nbl/builtin/glsl/utils/transform.glsl"

void main()
{
    const uint parent = nodeParent.data[iNodeID];
    vec3 pos = nodeGlobalTransforms.data[bool(gl_VertexIndex&0x1u)&&parent!=INVALID ? parent:iNodeID][3];
	// is there INVALID glsl keyword? Check it out, TODO!

    gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,pos);
	outColor = PushConstants.color;
}