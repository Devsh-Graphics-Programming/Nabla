#ifndef _IRR_BUILTIN_GLSL_BXDF_GEOM_SMITH_COMMON_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_GEOM_SMITH_COMMON_INCLUDED_

#include <irr/builtin/glsl/bxdf/ndf/common.glsl>

float irr_glsl_smith_G1(in float lambda)
{
    return 1.0 / (1.0 + lambda);
}

float irr_glsl_smith_VNDF_pdf_wo_clamps(in float ndf, in float lambda_V, in float maxNdotV, out float onePlusLambda_V)
{
    onePlusLambda_V = 1.0+lambda_V;

    return ndf*0.25/(maxNdotV*onePlusLambda_V);
}

float irr_glsl_smith_VNDF_pdf_wo_clamps(in float ndf, in float lambda_V, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta, in float reflectance, out float onePlusLambda_V)
{
    onePlusLambda_V = 1.0+lambda_V;

    return irr_glsl_microfacet_to_light_measure_transform((transmitted ? (1.0-reflectance):reflectance)*ndf/onePlusLambda_V,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta);
}

#endif
