#ifndef _IRR_BXDF_BRDF_SPECULAR_NDF_BLINN_PHONG_INCLUDED_
#define _IRR_BXDF_BRDF_SPECULAR_NDF_BLINN_PHONG_INCLUDED_

#include <irr/builtin/glsl/math/constants.glsl>

float irr_glsl_blinn_phong_ndf(in float NdotH, in float n)
{
    return irr_glsl_RECIPROCAL_PI*0.5*(n+2.0) * pow(NdotH,n);
}

#endif