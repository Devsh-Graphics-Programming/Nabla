#version 330 core

in vec4 Color; //per vertex output color, will be interpolated across the triangle

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = Color;
}
