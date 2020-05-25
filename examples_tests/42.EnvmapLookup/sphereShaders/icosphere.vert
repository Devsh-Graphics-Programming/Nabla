#version 430 core

layout(location = 0) in vec4 vPos;
layout(location = 2) in vec2 vUV;

#include <irr/builtin/glsl/vertex_utils/vertex_utils.glsl>

layout(set = 1, binding = 0, row_major, std140) uniform UBO
{
	irr_glsl_SBasicViewParameters params;
} cameraData;
 
layout(location = 0) out vec2 outUV;

void main()
{
	outUV = vUV;
    gl_Position = irr_glsl_pseudoMul4x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x4(cameraData.params.MVP), vPos.xyz);
}