#version 430 core
layout(binding=0) uniform sampler2D albedo;

in vec2 TexCoord;

layout(location = 0) out vec4 pixelColor;

void main()
{
	vec2 dUVdx = dFdx(TexCoord);
	vec2 dUVdy = dFdy(TexCoord);
    vec4 albedo_alpha = textureGrad(albedo,TexCoord,dUVdx,dUVdy);

	// alpha test, with TSSAA change it to stochastic test / alpha to weird coverage
	if (albedo_alpha.a<0.5)
		discard;

	pixelColor = albedo_alpha;
}
