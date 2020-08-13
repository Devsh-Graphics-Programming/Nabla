#ifndef _IRR_BXDF_BRDF_SPECULAR_NDF_BLINN_PHONG_INCLUDED_
#define _IRR_BXDF_BRDF_SPECULAR_NDF_BLINN_PHONG_INCLUDED_

#include <irr/builtin/glsl/math/constants.glsl>

float irr_glsl_blinn_phong(in float NdotH, in float n)
{
    return isinf(n) ? irr_glsl_FLT_INF : irr_glsl_RECIPROCAL_PI*0.5*(n+2.0) * pow(NdotH,n);
}
//ashikhmin-shirley ndf
float irr_glsl_blinn_phong(in float NdotH, in float one_minus_NdotH2_rcp, in float TdotH2, in float BdotH2, in float nx, in float ny)
{
    float n = (TdotH2*ny + BdotH2*nx) * one_minus_NdotH2_rcp;

    return (isinf(nx)||isinf(ny)) ?  irr_glsl_FLT_INF : sqrt((nx + 2.0)*(ny + 2.0))*irr_glsl_RECIPROCAL_PI*0.5 * pow(NdotH,n);
}

#endif