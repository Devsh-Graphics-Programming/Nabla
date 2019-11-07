#version 430 core
layout(location = 2) uniform vec3 color;
layout(location = 3) uniform uint nasty;

layout(binding = 0) uniform sampler2D reflectance;

in vec2 uv;
in vec3 Normal;

layout(location = 0) out vec3 pixelColor;
layout(location = 1) out vec2 encodedNormal;

#define MITS_TWO_SIDED		0x80000000u
#define MITS_USE_TEXTURE	0x40000000u


#define kPI 3.1415926536f
vec2 encode(in vec3 n)
{
    return vec2(atan(n.y,n.x)/kPI, n.z);
}
vec3 decode(in vec2 enc)
{
	float ang = enc.x*kPI;
    return vec3(vec2(cos(ang),sin(ang))*sqrt(1.0-enc.y*enc.y), enc.y);
}

void main()
{
	if ((nasty&MITS_USE_TEXTURE) == MITS_USE_TEXTURE)
	    pixelColor = texture(reflectance,uv).rgb;
	else
		pixelColor = color;

	encodedNormal = encode(normalize(Normal));
}
