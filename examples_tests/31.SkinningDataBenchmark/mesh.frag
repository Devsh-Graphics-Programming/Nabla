#version 330 core

in vec3 Normal;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = vec4(Normal*0.5f + vec3(0.5f), 1.f);
}
