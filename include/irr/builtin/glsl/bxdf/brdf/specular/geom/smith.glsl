#ifndef _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_
#define _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_

float irr_glsl_smith_beckmann_C2(in float NdotX2, in float a2)
{
    return NdotX2 / (a2 * (1.0 - NdotX2));
}
//G1 = 1/(1+_Lambda)
float irr_glsl_smith_beckmann_Lambda(in float c2)
{
    float c = sqrt(c2);
    float nom = 1.0 - 1.259*c + 0.396*c2;
    float denom = 2.181*c2 + 3.535*c;
    return mix(0.0, nom/denom, c<1.6);
}

float irr_glsl_smith_ggx_devsh_part(in float NdotX2, in float a2, in float one_minus_a2)
{
	return sqrt(a2+one_minus_a2*NdotX2);
}

float irr_glsl_GGXSmith_G1_(in float NdotX, in float a2, in float one_minus_a2)
{
    return (2.0*NdotX) / (NdotX + irr_glsl_smith_ggx_devsh_part(NdotX*NdotX,a2,one_minus_a2));
}
float irr_glsl_GGXSmith_G1_wo_numerator(in float NdotX, in float a2, in float one_minus_a2)
{
    return 1.0 / (NdotX + irr_glsl_smith_ggx_devsh_part(NdotX*NdotX,a2,one_minus_a2));
}

float irr_glsl_ggx_smith(in float a2, in float NdotL, in float NdotV)
{
	float one_minus_a2 = 1.0-a2;
    return irr_glsl_GGXSmith_G1_(NdotL,a2,one_minus_a2) * irr_glsl_GGXSmith_G1_(NdotV,a2,one_minus_a2);
}
float irr_glsl_ggx_smith_wo_numerator(in float a2, in float NdotL, in float NdotV)
{
	float one_minus_a2 = 1.0-a2;
    return irr_glsl_GGXSmith_G1_wo_numerator(NdotL,a2,one_minus_a2) * irr_glsl_GGXSmith_G1_wo_numerator(NdotV,a2,one_minus_a2);
}

float irr_glsl_ggx_smith_height_correlated_wo_numerator(in float a2, in float NdotL, in float NdotV)
{
	float one_minus_a2 = 1.0-a2;
    float denom = NdotV*irr_glsl_smith_ggx_devsh_part(NdotL*NdotL,a2,one_minus_a2) + NdotL*irr_glsl_smith_ggx_devsh_part(NdotV*NdotV,a2,one_minus_a2);
    return 0.5 / denom;
}

float irr_glsl_ggx_smith_height_correlated(in float a2, in float NdotL, in float NdotV)
{
    return 4.0*NdotL*NdotV*irr_glsl_ggx_smith_height_correlated_wo_numerator(a2,NdotL,NdotV);
}

// Note a, not a2!
float irr_glsl_ggx_smith_correlated_approx(in float a, in float NdotL, in float NdotV)
{
    float num = 2.0*NdotL*NdotV;
    return num / mix(num, NdotL+NdotV, a);
}

// Note a, not a2!
float irr_glsl_ggx_smith_correlated_approx_wo_numerator(in float a, in float NdotL, in float NdotV)
{
    return 0.5 / mix(2.0*NdotL*NdotV, NdotL+NdotV, a);
}

//depr
//Taken from https://google.github.io/filament/Filament.md.html#materialsystem/anisotropicmodel
float irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(in float at, in float ab, in float TdotL, in float TdotV, in float BdotL, in float BdotV, in float NdotL, in float NdotV)
{
    float Vterm = NdotL * length(vec3(at*TdotV, ab*BdotV, NdotV));
    float Lterm = NdotV * length(vec3(at*TdotL, ab*BdotL, NdotL));
    return 0.5 / (Vterm + Lterm);
}

float irr_glsl_ggx_smith_aniso_a0_2(in vec3 N, in vec3 X, in vec3 T, in float NdotX2, in float ax2, in float ay2)
{
    vec3 Xproj = normalize(X - dot(N,X)*N);
    float cos2phi = dot(T,Xproj);
    cos2phi *= cos2phi;
    float sin2phi = 1.0 - cos2phi;
    float a2 = cos2phi*ax2 + sin2phi*ay2;
}
float irr_glsl_ggx_smith_aniso_wo_numerator(in vec3 N, in vec3 L, in vec3 V, in vec3 T, in float NdotV, in float NdotL, in float ax2, in float ay2)
{
    float a02 = irr_glsl_ggx_smith_aniso_a0_2(N, V, T, NdotV*NdotV, ax2, ay2);
    float G1_V = irr_glsl_GGXSmith_G1_wo_numerator(NdotV, a02, 1.0-a02);
    a02 = irr_glsl_ggx_smith_aniso_a0_2(N, L, T, NdotL*NdotL, ax2, ay2);
    float G1_L = irr_glsl_GGXSmith_G1_wo_numerator(NdotL, a02, 1.0-a02);

    return G1_V*G1_L;
}
float irr_glsl_ggx_smith_aniso(in vec3 N, in vec3 L, in vec3 V, in vec3 T, in float NdotV, in float NdotL, in float ax2, in float ay2)
{
    return 4.0*NdotV*NdotL*irr_glsl_ggx_smith_aniso_wo_numerator(N, L, V, T, NdotV, NdotL, ax2, ay2);
}

float irr_glsl_ggx_smith_correlated_wo_numerator(in vec3 N, in vec3 L, in vec3 V, in vec3 T, in float NdotV, in float NdotL, in float ax2, in float ay2)
{
    float a02 = irr_glsl_ggx_smith_aniso_a0_2(N, L, T, NdotL*NdotL, ax2, ay2);
    float denom = NdotV*irr_glsl_smith_ggx_devsh_part(NdotL*NdotL,a02,1.0-a02);
	a02 = irr_glsl_ggx_smith_aniso_a0_2(N, V, T, NdotV*NdotV, ax2, ay2);
    denom += NdotL*irr_glsl_smith_ggx_devsh_part(NdotV*NdotV,a02,1.0-a02);
    return 0.5 / denom;
}
float irr_glsl_ggx_smith_correlated(in vec3 N, in vec3 L, in vec3 V, in vec3 T, in float NdotV, in float NdotL, in float ax2, in float ay2)
{
    return 4.0*NdotV*NdotL*irr_glsl_ggx_smith_correlated_wo_numerator(N, L, V, T, NdotV, NdotL, ax2, ay2);
}

#endif
