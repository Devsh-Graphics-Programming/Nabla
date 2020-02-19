#version 430 compatibility

#include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

layout(push_constant, row_major) uniform Block
{
	mat4 modelViewProjection;
	vec3 cameraPos;
} PushConstants;

layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec4 vCol;
layout(location = 3) in vec3 vNormal;

layout(location = 0) out vec4 Color; //per vertex output color, will be interpolated across the triangle
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec3 lightDir;

void main()
{
    gl_Position = irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4(PushConstants.modelViewProjection)*vPos; //only thing preventing the shader from being core-compliant
    Color = vCol;
    Normal = normalize(vNormal); //have to normalize twice because of normal quantization
    lightDir = cameraPos-vPos.xyz;
}
