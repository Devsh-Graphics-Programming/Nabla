#ifndef _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_
#define _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_

/* TODO remove
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
*/
float irr_glsl_smith_ggx_devsh_part(in float NdotX2, in float a2, in float one_minus_a2)
{
	return sqrt(a2+one_minus_a2*NdotX2)
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
    return 4.0*NdotL*NdotV*irr_glsl_ggx_smith_height_correlated_wo_numerator();
}

// Note a, not a2!
float irr_glsl_ggx_smith_height_correlated_approx(in float a, in float NdotL, in float NdotV)
{
    float num = 2.0*NdotL*NdotV;
    return num / mix(num, NdotL+NdotV, a);
}

// Note a, not a2!
float irr_glsl_ggx_smith_height_correlated_approx_wo_numerator(in float a, in float NdotL, in float NdotV)
{
    return 0.5 / mix(2.0*NdotL*NdotV, NdotL+NdotV, a);
}

//Taken from https://google.github.io/filament/Filament.md.html#materialsystem/anisotropicmodel
float irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(in float at, in float ab, in float TdotL, in float TdotV, in float BdotL, in float BdotV, in float NdotL, in float NdotV)
{
    float Vterm = NdotL * length(vec3(at*TdotV, ab*BdotV, NdotV));
    float Lterm = NdotV * length(vec3(at*TdotL, ab*BdotL, NdotL));
    return 0.5 / (Vterm + Lterm);
}

#endif
