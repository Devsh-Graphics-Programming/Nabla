#version 330 core
layout(location = 0) in vec4 vPos; //only a 3d position is passed from irrlicht, but last (the W) coordinate gets filled with default 1.0
layout(location = 1) in vec4 vCol;
layout(location = 3) in vec3 vNormal;

#include <irr/builtin/glsl/broken_driver_workarounds/amd.glsl>

//layout(row_major) uniform; not needed yet
uniform mat4 MVP;

out vec4 Color; //per vertex output color, will be interpolated across the triangle
out vec3 Normal;
out vec3 LocalPos;

void main()
{
    gl_Position = irr_builtin_glsl_workaround_AMD_broken_row_major_qualifier_mat4(MVP)*vPos; //only thing preventing the shader from being core-compliant
    Color = vCol;
    LocalPos = normalize(vPos.xyz);
    Normal = normalize(vNormal);
}
