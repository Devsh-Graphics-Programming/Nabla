#version 330 core

in vec4 Color; //per vertex output color, will be interpolated across the triangle
in vec3 Normal;
in vec3 LocalPos;

layout(location = 0) out vec4 pixelColor;

void main()
{
    //pixelColor = vec4(vec3(1.0-dot(normalize(Normal),normalize(LocalPos)))*10000.0,1.0);
    pixelColor = Color;
}
