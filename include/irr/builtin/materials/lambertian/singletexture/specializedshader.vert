#version 430 core

layout(location = 0) in vec3 vPos;
layout(location = 2) in vec2 vTexCoord;

#include <irr/builtin/glsl/utils/common.glsl>
#include <irr/builtin/glsl/utils/transform.glsl>
#include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

layout (set = 1, binding = 0, row_major, std140) uniform UBO 
{
    irr_glsl_SBasicViewParameters params;
} CamData;
        
layout(location = 0) out vec2 uv;

void main()
{
    gl_Position = irr_glsl_pseudoMul4x4with3x1(irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier(CamData.params.MVP), vPos);
	uv = vTexCoord;
}