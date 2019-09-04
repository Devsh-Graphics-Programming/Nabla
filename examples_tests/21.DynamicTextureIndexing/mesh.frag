#version 430 core

#define MAX_DRAWS 25
#define MAX_TEXTURES_PER_DRAW 1
layout(binding=0) uniform sampler2D textures[MAX_DRAWS][MAX_TEXTURES_PER_DRAW];

layout(location = 0) in flat uint DrawID;
layout(location = 1) in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;

void main()
{
	vec2 dUVdx = dFdx(TexCoord);
	vec2 dUVdy = dFdy(TexCoord);
    vec4 albedo_alpha = textureGrad(textures[DrawID][0],TexCoord,dUVdx,dUVdy);

	// alpha test, with TSSAA change it to stochastic test / alpha to weird coverage
	if (albedo_alpha.a<0.5)
		discard;

	pixelColor = albedo_alpha;
}
