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

layout(location = 0) out vec4 outColor;

#define INVALID 3735928559u
#include "nbl/builtin/glsl/utils/transform.glsl"

vec4 transform_(vec3 v)
{
	vec3 v2 = nbl_glsl_pseudoMul3x4with3x1(nodeGlobalTransforms.data[iNodeID],v * iScale);
	return nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,v2);
}

void main()
{
	if(gl_VertexIndex < 2) // render node-parent line
	{
		const uint parent = nodeParents.data[iNodeID];
		vec3 lineWorldPos = nodeGlobalTransforms.data[bool(gl_VertexIndex&0x1u)&&parent!=INVALID ? parent:iNodeID][3];

		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,lineWorldPos);
		outColor = PushConstants.lineColor;
	}
	else // render box
	{
		vec3 inGlobalTMinEdge = PushConstants.minEdge.xyz;
		vec3 inGlobalTMaxEdge = PushConstants.maxEdge.xyz;
		
		const vec4 vertex_base_0 = transform_(vec3(inGlobalTMinEdge.x, inGlobalTMinEdge.y, inGlobalTMinEdge.z));
		const vec4 vertex_base_1 = transform_(vec3(inGlobalTMinEdge.x, inGlobalTMinEdge.y, inGlobalTMaxEdge.z));
		const vec4 vertex_base_2 = transform_(vec3(inGlobalTMaxEdge.x, inGlobalTMinEdge.y, inGlobalTMinEdge.z));
		const vec4 vertex_base_3 = transform_(vec3(inGlobalTMaxEdge.x, inGlobalTMinEdge.y, inGlobalTMaxEdge.z));
		
		const vec4 vertex_ceiling_0 = transform_(vec3(inGlobalTMinEdge.x, inGlobalTMaxEdge.y, inGlobalTMinEdge.z));
		const vec4 vertex_ceiling_1 = transform_(vec3(inGlobalTMinEdge.x, inGlobalTMaxEdge.y, inGlobalTMaxEdge.z));
		const vec4 vertex_ceiling_2 = transform_(vec3(inGlobalTMaxEdge.x, inGlobalTMaxEdge.y, inGlobalTMinEdge.z));
		const vec4 vertex_ceiling_3 = transform_(vec3(inGlobalTMaxEdge.x, inGlobalTMaxEdge.y, inGlobalTMaxEdge.z));
		
		const vec4 boxVertices[] = 
		{
			//! null vertecies for shifting
			vec4(0, 0, 0, 0),
			vec4(0, 0, 0, 0),
			
			vertex_base_0,
			vertex_base_1,
			vertex_base_2,
			vertex_base_3,
			vertex_base_0,
			vertex_base_2,
			vertex_base_1,
			vertex_base_3,
			
			vertex_base_0,
			vertex_ceiling_0,
			vertex_base_1,
			vertex_ceiling_1,
			vertex_base_2,
			vertex_ceiling_2,
			vertex_base_3,
			vertex_ceiling_3,
			
			vertex_ceiling_0,
			vertex_ceiling_1,
			vertex_ceiling_2,
			vertex_ceiling_3,
			vertex_ceiling_0,
			vertex_ceiling_2,
			vertex_ceiling_1,
			vertex_ceiling_3
		};
		
		gl_Position = boxVertices[gl_VertexIndex];
		outColor = PushConstants.aabbColor;
	}
}