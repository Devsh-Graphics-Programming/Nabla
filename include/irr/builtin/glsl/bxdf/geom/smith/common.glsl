#ifndef _NBL_BUILTIN_GLSL_BXDF_GEOM_SMITH_COMMON_INCLUDED_
#define _NBL_BUILTIN_GLSL_BXDF_GEOM_SMITH_COMMON_INCLUDED_

#include <irr/builtin/glsl/bxdf/ndf/common.glsl>

float nbl_glsl_smith_G1(in float lambda)
{
    return 1.0 / (1.0 + lambda);
}



float nbl_glsl_smith_VNDF_pdf_wo_clamps(in float ndf, in float lambda_V, in float maxNdotV, out float onePlusLambda_V)
{
    onePlusLambda_V = 1.0+lambda_V;

    return nbl_glsl_microfacet_to_light_measure_transform(ndf/onePlusLambda_V,maxNdotV);
}

float nbl_glsl_smith_VNDF_pdf_wo_clamps(in float ndf, in float lambda_V, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta, in float reflectance, out float onePlusLambda_V)
{
    onePlusLambda_V = 1.0+lambda_V;

    return nbl_glsl_microfacet_to_light_measure_transform((transmitted ? (1.0-reflectance):reflectance)*ndf/onePlusLambda_V,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta);
}


float nbl_glsl_smith_VNDF_pdf_wo_clamps(in float ndf, in float G1_over_2NdotV)
{
    return ndf*0.5*G1_over_2NdotV;
}

float nbl_glsl_smith_FVNDF_pdf_wo_clamps(in float fresnel_ndf, in float G1_over_2NdotV, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
{
    float FNG = fresnel_ndf * G1_over_2NdotV;
    float factor = 0.5;
    if (transmitted)
    {
        const float VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
        // VdotHLdotH is negative under transmission, so this factor is negative
        factor *= -2.0 * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
    }
    return FNG * factor;
}

float nbl_glsl_smith_VNDF_pdf_wo_clamps(in float ndf, in float G1_over_2NdotV, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta, in float reflectance)
{
    float FN = (transmitted ? (1.0 - reflectance) : reflectance) * ndf;
    
    return nbl_glsl_smith_FVNDF_pdf_wo_clamps(FN, G1_over_2NdotV, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
}

#endif
