#version 330 core
uniform sampler2D tex0;

in vec3 Normal;
in vec2 TexCoord;
in vec3 lightDir;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = texture(tex0,TexCoord)*max(dot(normalize(Normal),normalize(lightDir)),0.0);
}
