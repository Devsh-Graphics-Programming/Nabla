

// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_GEOM_SMITH_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_GEOM_SMITH_COMMON_INCLUDED_


#include <nbl/builtin/hlsl/bxdf/ndf.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace geom_smith
{


float G1(in float lambda)
{
    return 1.0 / (1.0 + lambda);
}

float G2(in float lambda_V, in float lambda_L)
{
    return 1.0 / (1.0 + lambda_V + lambda_L);
}


float VNDF_pdf_wo_clamps(in float ndf, in float lambda_V, in float maxNdotV, out float onePlusLambda_V)
{
    onePlusLambda_V = 1.0+lambda_V;

    return ndf::microfacet_to_light_measure_transform(ndf/onePlusLambda_V,maxNdotV);
}

float VNDF_pdf_wo_clamps(in float ndf, in float lambda_V, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta, in float reflectance, out float onePlusLambda_V)
{
    onePlusLambda_V = 1.0+lambda_V;

    return ndf::microfacet_to_light_measure_transform((transmitted ? (1.0-reflectance):reflectance)*ndf/onePlusLambda_V,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta);
}

// for when you know the NDF and the uncorrelated smith masking function
float VNDF_pdf_wo_clamps(in float ndf, in float G1_over_2NdotV)
{
    return ndf*0.5*G1_over_2NdotV;
}

float FVNDF_pdf_wo_clamps(in float fresnel_ndf, in float G1_over_2NdotV, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
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

float VNDF_pdf_wo_clamps(in float ndf, in float G1_over_2NdotV, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta, in float reflectance)
{
    float FN = (transmitted ? (1.0 - reflectance) : reflectance) * ndf;
    
    return FVNDF_pdf_wo_clamps(FN, G1_over_2NdotV, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
}

	
}
}
}
}


#endif