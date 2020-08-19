#ifndef _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_
#define _BRDF_SPECULAR_GEOM_GGX_SMITH_INCLUDED_

//TODO divide into more files, one for smith for each NDF

float irr_glsl_smith_G1(in float lambda)
{
    return 1.0 / (1.0 + lambda);
}

//GGX

float irr_glsl_smith_ggx_devsh_part(in float NdotX2, in float a2, in float one_minus_a2)
{
    return sqrt(a2+one_minus_a2*NdotX2);
}
float irr_glsl_smith_ggx_devsh_part(in float TdotX2, in float BdotX2, in float NdotX2, in float ax2, in float ay2)
{
    return sqrt(TdotX2*ax2+BdotX2*ay2+NdotX2);
}

float irr_glsl_GGXSmith_G1_wo_numerator(in float NdotX, in float NdotX2, in float a2, in float one_minus_a2)
{
    return 1.0 / (NdotX + irr_glsl_smith_ggx_devsh_part(NdotX2,a2,one_minus_a2));
}
float irr_glsl_GGXSmith_G1_wo_numerator(in float NdotX, in float TdotX2, in float BdotX2, in float NdotX2, in float ax2, in float ay2)
{
    return 1.0 / (NdotX + irr_glsl_smith_ggx_devsh_part(TdotX2, BdotX2, NdotX2, ax2, ay2));
}
float irr_glsl_GGXSmith_G1_wo_numerator(in float NdotX, in float devsh_part)
{
    return 1.0 / (NdotX + devsh_part);
}

float irr_glsl_ggx_smith_correlated_wo_numerator(in float NdotV, in float NdotV2, in float NdotL, in float NdotL2, in float a2, in float one_minus_a2)
{
    float Vterm = NdotL*irr_glsl_smith_ggx_devsh_part(NdotV2,a2,one_minus_a2);
    float Lterm = NdotV*irr_glsl_smith_ggx_devsh_part(NdotL2,a2,one_minus_a2);
    return 0.5 / (Vterm + Lterm);
}
/* depr
float irr_glsl_ggx_smith_correlated(in float NdotV, in float NdotV2, in float NdotL, in float NdotL2, in float a2, in float one_minus_a2)
{
    return 4.0*NdotV*NdotL*irr_glsl_ggx_smith_correlated_wo_numerator(NdotV, NdotV2, NdotL, NdotL2, a2, one_minus_a2);
}
*/
float irr_glsl_ggx_smith_correlated_wo_numerator(in float NdotV, in float NdotV2, in float NdotL, in float NdotL2, in float a2)
{
    return irr_glsl_ggx_smith_correlated_wo_numerator(NdotV,NdotV2,NdotL,NdotL2,a2,1.0-a2);
}
/* depr
float irr_glsl_ggx_smith_correlated(in float NdotV, in float NdotV2, in float NdotL, in float NdotL2, in float a2)
{
    return 4.0*NdotV*NdotL*irr_glsl_ggx_smith_correlated_wo_numerator(NdotV, NdotV2, NdotL, NdotL2, a2);
}
*/
float irr_glsl_ggx_smith_correlated_wo_numerator(in float NdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float NdotL, in float TdotL2, in float BdotL2, in float NdotL2, in float ax2, in float ay2)
{
    float Vterm = NdotL*irr_glsl_smith_ggx_devsh_part(TdotV2,BdotV2,NdotV2,ax2,ay2);
    float Lterm = NdotV*irr_glsl_smith_ggx_devsh_part(TdotL2,BdotL2,NdotL2,ax2,ay2);
    return 0.5 / (Vterm + Lterm);
}
/* depr
float irr_glsl_ggx_smith_correlated(in float NdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float NdotL, in float TdotL2, in float BdotL2, in float NdotL2, in float ax2, in float ay2)
{
    return 4.0*NdotV*NdotL*irr_glsl_ggx_smith_correlated_wo_numerator(NdotV, TdotV2, BdotV2, NdotV2, NdotL, TdotL2, BdotL2, NdotL2, ax2, ay2);
}
*/

float irr_glsl_ggx_smith_G2_over_G1(in float NdotL, in float NdotL2, in float NdotV, in float NdotV2, in float a2, in float one_minus_a2)
{
    float devsh_v = irr_glsl_smith_ggx_devsh_part(NdotV2,a2,one_minus_a2);
	float G2_over_G1 = NdotL*(devsh_v + NdotV); // alternative `Vterm+NdotL*NdotV /// NdotL*NdotV could come as a parameter
	G2_over_G1 /= NdotV*irr_glsl_smith_ggx_devsh_part(NdotL2,a2,one_minus_a2) + NdotL*devsh_v;

    return G2_over_G1;
}
float irr_glsl_ggx_smith_G2_over_G1_devsh(in float NdotL, in float NdotL2, in float NdotV, in float devsh_v, in float a2, in float one_minus_a2)
{
	float G2_over_G1 = NdotL*(devsh_v + NdotV); // alternative `Vterm+NdotL*NdotV /// NdotL*NdotV could come as a parameter
	G2_over_G1 /= NdotV*irr_glsl_smith_ggx_devsh_part(NdotL2,a2,one_minus_a2) + NdotL*devsh_v;

    return G2_over_G1;
}
float irr_glsl_ggx_smith_G2_over_G1(in float NdotL, in float TdotL2, in float BdotL2, in float NdotL2, in float NdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float ax2, in float ay2)
{
    float devsh_v = irr_glsl_smith_ggx_devsh_part(TdotV2,BdotV2,NdotV2,ax2,ay2);
	float G2_over_G1 = NdotL*(devsh_v + NdotV);
	G2_over_G1 /= NdotV*irr_glsl_smith_ggx_devsh_part(TdotL2,BdotL2,NdotL2,ax2,ay2) + NdotL*devsh_v;

    return G2_over_G1;
}
float irr_glsl_ggx_smith_G2_over_G1(in float NdotL, in float TdotL2, in float BdotL2, in float NdotL2, in float NdotV, in float devsh_v, in float ax2, in float ay2)
{
	float G2_over_G1 = NdotL*(devsh_v + NdotV);
	G2_over_G1 /= NdotV*irr_glsl_smith_ggx_devsh_part(TdotL2,BdotL2,NdotL2,ax2,ay2) + NdotL*devsh_v;

    return G2_over_G1;
}

//Beckmann

float irr_glsl_smith_beckmann_C2(in float NdotX2, in float a2)
{
    return NdotX2 / (a2 * (1.0 - NdotX2));
}
float irr_glsl_smith_beckmann_C2(in float TdotX2, in float BdotX2, in float NdotX2, in float ax2, in float ay2)
{
    return NdotX2/(TdotX2*ax2+BdotX2*ay2);
}
//G1 = 1/(1+_Lambda)
float irr_glsl_smith_beckmann_Lambda(in float c2)
{
    float c = sqrt(c2);
    float nom = 1.0 - 1.259*c + 0.396*c2;
    float denom = 2.181*c2 + 3.535*c;
    return mix(0.0, nom/denom, c<1.6);
}
float irr_glsl_smith_beckmann_Lambda(in float NdotX2, in float a2)
{
    return irr_glsl_smith_beckmann_Lambda(irr_glsl_smith_beckmann_C2(NdotX2, a2));
}

float irr_glsl_beckmann_smith_correlated(in float NdotV2, in float NdotL2, in float a2)
{
    float c2 = irr_glsl_smith_beckmann_C2(NdotV2, a2);
    float L_v = irr_glsl_smith_beckmann_Lambda(c2);
    c2 = irr_glsl_smith_beckmann_C2(NdotL2, a2);
    float L_l = irr_glsl_smith_beckmann_Lambda(c2);
    return 1.0 / (1.0 + L_v + L_l);
}
float irr_glsl_beckmann_smith_correlated(in float TdotV2, in float BdotV2, in float NdotV2, in float TdotL2, in float BdotL2, in float NdotL2, in float ax2, in float ay2)
{
    float c2 = irr_glsl_smith_beckmann_C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
    float L_v = irr_glsl_smith_beckmann_Lambda(c2);
    c2 = irr_glsl_smith_beckmann_C2(TdotL2, BdotL2, NdotL2, ax2, ay2);
    float L_l = irr_glsl_smith_beckmann_Lambda(c2);
    return 1.0 / (1.0 + L_v + L_l);
}

float irr_glsl_beckmann_smith_G2_over_G1(in float lambdaV_plus_one, in float NdotL, in float NdotL2, in float a2)
{
    float lambdaL = irr_glsl_smith_beckmann_Lambda(NdotL2, a2);

    return NdotL*lambdaV_plus_one / (lambdaV_plus_one+lambdaL);
}
float irr_glsl_beckmann_smith_G2_over_G1(in float lambdaV_plus_one, in float NdotL, in float TdotL2, in float BdotL2, in float NdotL2, in float ax2, in float ay2)
{
    float c2 = irr_glsl_smith_beckmann_C2(TdotL2, BdotL2, NdotL2, ax2, ay2);
	float lambdaL = irr_glsl_smith_beckmann_Lambda(c2);

    return NdotL*lambdaV_plus_one / (lambdaV_plus_one + lambdaL);
}

#endif
