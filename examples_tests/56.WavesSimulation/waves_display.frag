#version 430 core
layout(set = 3, binding = 0) uniform sampler2D tex0;

layout(location = 0) in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;

void main()
{
    vec2 tex_coord = vec2(TexCoord.x, 1 - TexCoord.y); 
    pixelColor = vec4(textureLod(tex0, tex_coord, 0.0).rgb, 1);
}