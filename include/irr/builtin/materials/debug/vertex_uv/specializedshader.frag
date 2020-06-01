#version 430 core

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = vec4(uv.x, uv.y, 1.0, 1.0);
}