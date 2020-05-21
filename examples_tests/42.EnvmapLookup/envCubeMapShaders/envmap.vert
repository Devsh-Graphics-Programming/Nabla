#version 430 core

layout(location = 0) in vec4 vPos;
layout(location = 3) in vec3 vNormal;

#include <irr/builtin/glsl/vertex_utils/vertex_utils.glsl>

layout(set = 1, binding = 0, row_major, std140) uniform UBO
{
	irr_glsl_SBasicViewParameters params;
} cameraData;

layout(location = 0) out vec3 localCubePosition; 

void main()
{
	localCubePosition = vPos.xyz;
    gl_Position = irr_glsl_pseudoMul4x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4x4(cameraData.params.MVP), localCubePosition);
}