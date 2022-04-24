#version 430 core

layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNorm;

layout(push_constant, row_major) uniform PushConstants
{
    mat4 vp;
} pc;

layout(location = 0) out vec3 normal;

void main()
{
    gl_Position = pc.vp * vec4(vPos, 1.0);
    normal = normalize(vNorm);
}