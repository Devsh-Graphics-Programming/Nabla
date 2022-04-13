#version 430 core

layout(location = 0) in vec3 normal;
layout(location = 0) out vec4 color;

void main()
{
    vec3 colorTmp = clamp(dot(vec3(0.0, 1.0, 0.0), normal), 0, 1) * vec3(1.0) + vec3(0.2);
    color = vec4(colorTmp, 1.0);
}