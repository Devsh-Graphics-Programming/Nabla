#ifndef _IRR_BSDF_BRDF_SPECULAR_NDF_ASHIKHMIN_SHIRLEY_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_NDF_ASHIKHMIN_SHIRLEY_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>

//n is 2 phong-like exponents for anisotropy, can be defined as vec2(1.0/at, 1.0/ab) where at is roughness along tangent direction and ab is roughness along bitangent direction
//sin_cos_phi is sin and cos of azimuth angle of half vector
float irr_glsl_ashikhmin_shirley(in float NdotL, in float NdotV, in float NdotH, in float VdotH, in vec2 n, in vec2 sin_cos_phi)
{
    float nom = sqrt((n.x + 1.0)*(n.y + 1.0)) * pow(NdotH, n.x*sin_cos_phi.x*sin_cos_phi.x + n.y*sin_cos_phi.y*sin_cos_phi.y);
    float denom = 8.0 * irr_glsl_PI * VdotH * max(NdotV,NdotL);

    return NdotL * nom/denom;
}

#endif
