#version 430 core

layout(location = 0) in vec4 vPosition; 
layout(location = 1) in vec4 vVelocity;
layout(location = 2) in vec4 vColor;

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>
#include <nbl/builtin/glsl/broken_driver_workarounds/amd.glsl>

layout (set = 1, binding = 0, row_major, std140) uniform UBO 
{
    nbl_glsl_SBasicViewParameters params;
} cameraData;

layout(location = 0) flat out vec4 outGOrFFullyProjectedVelocity;
layout(location = 1) flat out vec4 outGorFColor;

void main()
{
    gl_Position = nbl_builtin_glsl_workaround_AMD_broken_row_major_qualifier(cameraData.params.MVP) * vPosition;
    outGOrFFullyProjectedVelocity = nbl_builtin_glsl_workaround_AMD_broken_row_major_qualifier(cameraData.params.MVP) * vVelocity * 0.0001;
    outGorFColor = vColor;
}