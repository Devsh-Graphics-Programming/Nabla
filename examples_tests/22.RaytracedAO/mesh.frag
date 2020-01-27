#version 430 core
layout(location = 2) uniform vec3 color;
layout(location = 3) uniform uint nasty;
layout(location = 4) uniform uint lightID;

layout(binding = 0) uniform sampler2D reflectance;

in vec2 uv;
in vec3 Normal;

layout(location = 0) out vec3 pixelColor;
layout(location = 1) out vec2 encodedNormal;
layout(location = 2) out uint lightIndex;

#define MITS_TWO_SIDED		0x80000000u
#define MITS_USE_TEXTURE	0x40000000u


#define kPI 3.1415926536f
vec2 encode(in vec3 n)
{
    return vec2(atan(n.y,n.x)/kPI, n.z);
}

void main()
{
	if ((nasty&MITS_USE_TEXTURE) == MITS_USE_TEXTURE)
	    pixelColor = texture(reflectance,uv).rgb;
	else
		pixelColor = color;

	encodedNormal = encode(normalize(Normal));
	lightIndex = lightID;
}
