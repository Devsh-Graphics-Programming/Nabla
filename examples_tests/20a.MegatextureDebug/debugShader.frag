#version 430 core

layout (set = 0, binding = 1) uniform sampler2DArray tex[3];
layout (set = 0, binding = 0) uniform usampler2D pgtab[3];

layout (location = 0) in vec2 TexCoord;
layout (location = 0) out vec4 OutColor;

void main()
{
    uvec4 pg = texelFetch(pgtab[1], ivec2(0,4), 0);
    vec3 col = pg.x!=(0xffffffffu-1u) ? texture(tex[1], vec3(TexCoord, 0.0)).rgb : vec3(1.0);
    OutColor = vec4(col, 1.0);
}