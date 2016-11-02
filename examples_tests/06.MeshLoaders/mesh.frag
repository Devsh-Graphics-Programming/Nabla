#version 330 core

in vec4 Color; //per vertex output color, will be interpolated across the triangle
in vec3 Normal;
in vec3 lightDir;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = Color*max(dot(normalize(Normal),normalize(lightDir)),0.0);
}
