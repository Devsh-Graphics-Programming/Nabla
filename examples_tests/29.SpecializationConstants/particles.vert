#version 430 core

layout (location = 0) in vec3 vPos;

#include <irr/builtin/glsl/utils/common.glsl>
#include <irr/builtin/glsl/utils/transform.glsl>

layout (set = 0, binding = 0, row_major, std140) uniform UBO
{
    irr_glsl_SBasicViewParameters params;
} CamData;

void main()
{
	gl_Position = irr_glsl_pseudoMul4x4with3x1(CamData.params.MVP, vPos);
}