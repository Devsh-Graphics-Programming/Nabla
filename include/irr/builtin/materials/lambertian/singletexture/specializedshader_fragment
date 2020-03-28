#version 430 core

layout(set = 0, binding = 0) uniform sampler2D albedo;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 pixelColor;

void main()
{
    pixelColor = texture(albedo,uv);
}
		