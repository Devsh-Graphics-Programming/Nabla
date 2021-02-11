#version 460 core

layout (location = 0) in vec3 LocalPos;
layout (location = 1) in vec3 ViewPos;
layout (location = 2) in vec3 Normal;
layout (location = 3) in vec2 UV;

layout(location = 0) out vec4 color;

void main()
{
    vec3 colorTmp = clamp(dot(vec3(0.0, 1.0, 0.0), Normal), 0, 1) * vec3(1.0) + vec3(0.2);
    color = vec4(colorTmp, 1.0);

    //color = vec4(1.0);
}