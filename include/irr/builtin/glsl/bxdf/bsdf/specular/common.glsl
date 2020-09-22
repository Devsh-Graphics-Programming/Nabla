#ifndef _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_COMMON_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_COMMON_INCLUDED_

#include <irr/builtin/glsl/math/functions.glsl>

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/fresnel.glsl>


float irr_glsl_smith_VNDF_pdf_wo_clamps(in float ndf, in float lambda_V, in float absNdotV, in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta, in float reflectance, out float onePlusLambda_V)
{
    onePlusLambda_V = 1.0+lambda_V;

    float denominator = absNdotV*onePlusLambda_V;
    if (transmitted)
    {
        const float VdotH_etaLdotH = (VdotH+orientedEta*LdotH);
        denominator *= VdotH_etaLdotH*VdotH_etaLdotH;
    }
    // VdotHLdotH is negative under transmission, so thats why fresnel transmission has a negative form
    return ndf*(transmitted ? VdotHLdotH:0.25)*(transmitted ? (reflectance-1.0):reflectance)/denominator;
}

#endif