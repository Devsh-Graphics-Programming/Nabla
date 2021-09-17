#version 460 core

layout(location = 0) in vec3 inGlobalTMinEdge[];
layout(location = 1) in vec3 inGlobalTMaxEdge[];

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
layout(line_strip, max_vertices = 36) out;

void main()
{
		//! render line
		outColor = PushConstants.lineColor;
		gl_Position = gl_in[0].gl_Position;
		EmitVertex();
		gl_Position = gl_in[1].gl_Position;
		EmitVertex();
		EndPrimitive();

		outColor = PushConstants.aabbColor;
		
		const vec3 vertex_base_0 = vec3(inGlobalTMinEdge[0].x, inGlobalTMinEdge[0].y, inGlobalTMinEdge[0].z);
		const vec3 vertex_base_1 = vec3(inGlobalTMinEdge[0].x, inGlobalTMinEdge[0].y, inGlobalTMaxEdge[0].z);
		const vec3 vertex_base_2 = vec3(inGlobalTMaxEdge[0].x, inGlobalTMinEdge[0].y, inGlobalTMinEdge[0].z);
		const vec3 vertex_base_3 = vec3(inGlobalTMaxEdge[0].x, inGlobalTMinEdge[0].y, inGlobalTMaxEdge[0].z);
		
		const vec3 vertex_ceiling_0 = vec3(inGlobalTMinEdge[0].x, inGlobalTMaxEdge[0].y, inGlobalTMinEdge[0].z);
		const vec3 vertex_ceiling_1 = vec3(inGlobalTMinEdge[0].x, inGlobalTMaxEdge[0].y, inGlobalTMaxEdge[0].z);
		const vec3 vertex_ceiling_2 = vec3(inGlobalTMaxEdge[0].x, inGlobalTMaxEdge[0].y, inGlobalTMinEdge[0].z);
		const vec3 vertex_ceiling_3 = vec3(inGlobalTMaxEdge[0].x, inGlobalTMaxEdge[0].y, inGlobalTMaxEdge[0].z);
		
		//! render box
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_base_0);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_base_1);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_base_3);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_base_2);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_base_0);
		EmitVertex();
		EndPrimitive();
		
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_ceiling_0);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_ceiling_1);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_ceiling_3);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_ceiling_2);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_ceiling_0);
		EmitVertex();
		EndPrimitive();
		
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_base_0);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_ceiling_0);
		EmitVertex();
		EndPrimitive();
		
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_base_1);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_ceiling_1);
		EmitVertex();
		EndPrimitive();
		
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_base_2);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_ceiling_2);
		EmitVertex();
		EndPrimitive();
		
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_base_3);
		EmitVertex();
		gl_Position = nbl_glsl_pseudoMul4x4with3x1(PushConstants.viewProj,vertex_ceiling_3);
		EmitVertex();
		EndPrimitive();
}