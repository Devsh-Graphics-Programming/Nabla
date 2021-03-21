#version 460 core

layout(push_constant, row_major) uniform PushConstants
{
	mat4 vp;
} pc;

layout(location = 0) in vec3 vPos;
layout(location = 3) in vec3 vNorm;

layout(location = 15) in vec3 vTransposition;

layout(location = 0) out vec3 normal;

void main()
{
    vec3 pos = vPos + vTransposition;
    gl_Position = pc.vp * vec4(pos, 1.0);
    normal = normalize(vNorm);
}