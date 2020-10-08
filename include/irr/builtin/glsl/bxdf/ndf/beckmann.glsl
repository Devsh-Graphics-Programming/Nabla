#ifndef _IRR_BSDF_BECKMANN_INCLUDED_
#define _IRR_BSDF_BECKMANN_INCLUDED_

#include <irr/builtin/glsl/math/constants.glsl>
#include <irr/builtin/glsl/bxdf/ndf/common.glsl>

float irr_glsl_beckmann(in float a2, in float NdotH2)
{
    float nom = exp( (NdotH2-1.0)/(a2*NdotH2) ); // exp(x) == exp2(x/log(2)) ?
    float denom = a2*NdotH2*NdotH2;

    return irr_glsl_RECIPROCAL_PI * nom/denom;
}

float irr_glsl_beckmann(in float ax, in float ay, in float ax2, in float ay2, in float TdotH2, in float BdotH2, in float NdotH2)
{
    float nom = exp(-(TdotH2/ax2+BdotH2/ay2)/NdotH2);
    float denom = ax * ay * NdotH2 * NdotH2;

    return irr_glsl_RECIPROCAL_PI * nom / denom;
}


#endif
