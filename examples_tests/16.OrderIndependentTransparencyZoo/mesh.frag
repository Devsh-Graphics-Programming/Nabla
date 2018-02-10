#version 420 core
layout(early_fragment_tests) in;

uniform sampler2D tex0;
uniform vec3 selfPos;

in vec2 TexCoord;
in float height;

layout(location = 0) out vec4 pixelColor;

void main()
{
    float alpha = min(height*(length(selfPos.xz)*0.02+1.0)*0.03,1.0);
    pixelColor = vec4(texture(tex0,TexCoord).rgb,alpha);
}
