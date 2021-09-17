#version 460 core

layout(location = 0) in float inIScale[];
layout(location = 1) in mat4x3 inNodeGlobalTransform[];

layout( push_constant, row_major ) uniform Block
{
    mat4 viewProj;
	vec4 lineColor;
	vec4 aabbColor;
	vec4 minEdge;
	vec4 maxEdge;
} PushConstants;

layout(location = 0) out vec4 outColor;

#include "nbl/builtin/glsl/utils/transform.glsl"

layout(lines) in;
layout(line_strip, max_vertices = 40) out;

vec4 transform_(vec3 v)
{
	vec3 v2 = nbl_glsl_pseudoMul3x4with3x1(inNodeGlobalTransform[0],v * inIScale[0]);
	return nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,v2);
}

void main()
{
		vec3 inGlobalTMinEdge = PushConstants.minEdge.xyz;
		vec3 inGlobalTMaxEdge = PushConstants.maxEdge.xyz;

		//! render line
		outColor = PushConstants.lineColor;
		gl_Position = gl_in[0].gl_Position;
		EmitVertex();
		gl_Position = gl_in[1].gl_Position;
		EmitVertex();
		EndPrimitive();

		outColor = PushConstants.aabbColor;
		
		const vec4 vertex_base_0 = transform_(vec3(inGlobalTMinEdge.x, inGlobalTMinEdge.y, inGlobalTMinEdge.z));
		const vec4 vertex_base_1 = transform_(vec3(inGlobalTMinEdge.x, inGlobalTMinEdge.y, inGlobalTMaxEdge.z));
		const vec4 vertex_base_2 = transform_(vec3(inGlobalTMaxEdge.x, inGlobalTMinEdge.y, inGlobalTMinEdge.z));
		const vec4 vertex_base_3 = transform_(vec3(inGlobalTMaxEdge.x, inGlobalTMinEdge.y, inGlobalTMaxEdge.z));
		
		const vec4 vertex_ceiling_0 = transform_(vec3(inGlobalTMinEdge.x, inGlobalTMaxEdge.y, inGlobalTMinEdge.z));
		const vec4 vertex_ceiling_1 = transform_(vec3(inGlobalTMinEdge.x, inGlobalTMaxEdge.y, inGlobalTMaxEdge.z));
		const vec4 vertex_ceiling_2 = transform_(vec3(inGlobalTMaxEdge.x, inGlobalTMaxEdge.y, inGlobalTMinEdge.z));
		const vec4 vertex_ceiling_3 = transform_(vec3(inGlobalTMaxEdge.x, inGlobalTMaxEdge.y, inGlobalTMaxEdge.z));
		
		//! render box
		gl_Position = vertex_base_0;
		EmitVertex();
		gl_Position = vertex_base_1;
		EmitVertex();
		gl_Position = vertex_base_3;
		EmitVertex();
		gl_Position = vertex_base_2;
		EmitVertex();
		gl_Position = vertex_base_0;
		EmitVertex();
		EndPrimitive();
		
		gl_Position = vertex_ceiling_0;
		EmitVertex();
		gl_Position = vertex_ceiling_1;
		EmitVertex();
		gl_Position = vertex_ceiling_3;
		EmitVertex();
		gl_Position = vertex_ceiling_2;
		EmitVertex();
		gl_Position = vertex_ceiling_0;
		EmitVertex();
		EndPrimitive();
		
		gl_Position = vertex_base_0;
		EmitVertex();
		gl_Position = vertex_ceiling_0;
		EmitVertex();
		EndPrimitive();
		
		gl_Position = vertex_base_1;
		EmitVertex();
		gl_Position = vertex_ceiling_1;
		EmitVertex();
		EndPrimitive();
		
		gl_Position = vertex_base_2;
		EmitVertex();
		gl_Position = vertex_ceiling_2;
		EmitVertex();
		EndPrimitive();
		
		gl_Position = vertex_base_3;
		EmitVertex();
		gl_Position = vertex_ceiling_3;
		EmitVertex();
		EndPrimitive();
}