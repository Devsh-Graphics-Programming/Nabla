#version 430 core

layout(location = 0) in vec4 vPosition; 
layout(location = 1) in vec4 vVelocity;
layout(location = 2) in vec4 vColor;

#include <nbl/builtin/glsl/utils/common.glsl>
#include <nbl/builtin/glsl/utils/transform.glsl>

layout (set = 1, binding = 0, row_major, std140) uniform UBO 
{
    nbl_glsl_SBasicViewParameters params;
} cameraData;

layout(location = 0) flat out vec4 outGOrFFullyProjectedVelocity;
layout(location = 1) flat out vec4 outGorFColor;

void main()
{
    gl_Position = (cameraData.params.MVP) * vPosition;
    outGOrFFullyProjectedVelocity = (cameraData.params.MVP) * vVelocity * 0.0001;
    outGorFColor = vColor;
}