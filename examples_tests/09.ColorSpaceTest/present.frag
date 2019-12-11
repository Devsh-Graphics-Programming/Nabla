#version 430 core
layout(set = 0, binding = 0) uniform sampler2D tex0;

layout(location = 0) in vec2 TexCoord;


layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = textureLod(tex0,TexCoord,0.0);
}

