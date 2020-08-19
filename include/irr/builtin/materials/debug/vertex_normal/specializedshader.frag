#version 430 core
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec3 color; 
layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = vec4(color,1.0);
}