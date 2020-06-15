#ifndef _IRR_BSDF_BRDF_SPECULAR_BECKMANN_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_BECKMANN_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

float irr_glsl_beckmann(in float a2, in float NdotH2)
{
    float nom = exp( (NdotH2 - 1.0)/(a2*NdotH2) );
    float denom = a2*NdotH2*NdotH2;

    return irr_glsl_RECIPROCAL_PI * nom/denom;
}

#endif
