#version 450 core

layout (location = 0) in vec4 Color; //per vertex output color, will be interpolated across the triangle
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec3 lightDir;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = Color*max(dot(normalize(Normal),normalize(lightDir)),0.0);
}
