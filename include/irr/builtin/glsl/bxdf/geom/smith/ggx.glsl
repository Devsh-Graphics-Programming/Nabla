#ifndef _IRR_BXDF_GEOM_SMITH_GGX_INCLUDED_
#define _IRR_BXDF_GEOM_SMITH_GGX_INCLUDED_


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
float irr_glsl_ggx_smith_G2_over_G1_devsh(in float NdotL, in float TdotL2, in float BdotL2, in float NdotL2, in float NdotV, in float devsh_v, in float ax2, in float ay2)
{
	float G2_over_G1 = NdotL*(devsh_v + NdotV);
	G2_over_G1 /= NdotV*irr_glsl_smith_ggx_devsh_part(TdotL2,BdotL2,NdotL2,ax2,ay2) + NdotL*devsh_v;

    return G2_over_G1;
}

#endif
