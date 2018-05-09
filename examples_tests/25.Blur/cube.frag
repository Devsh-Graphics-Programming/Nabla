#version 430 core

in vec3 Normal;
in vec3 lightDir;
in vec2 uv;

layout(location = 0) out vec4 pixelColor;

layout(binding = 0) uniform sampler2D tex;

void main()
{
    pixelColor = texture(tex, uv) * max(dot(Normal, normalize(lightDir)), 0.);
}
