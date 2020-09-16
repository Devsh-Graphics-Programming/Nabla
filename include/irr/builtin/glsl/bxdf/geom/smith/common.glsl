#ifndef _IRR_BXDF_GEOM_SMITH_COMMON_INCLUDED_
#define _IRR_BXDF_GEOM_SMITH_COMMON_INCLUDED_

float irr_glsl_smith_G1(in float lambda)
{
    return 1.0 / (1.0 + lambda);
}

float irr_glsl_smith_VNDF_pdf_wo_clamps(in float ndf, in float lambda_V, in float maxNdotV, out float onePlusLambda_V)
{
    onePlusLambda_V = 1.0+lambda_V;
    const float G1 = 1.0/onePlusLambda_V;

    return ndf*G1*0.25/maxNdotV;
}

#endif
