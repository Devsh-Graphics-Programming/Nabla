#ifndef _IRR_BSDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_

#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/fresnel/fresnel.glsl>

float irr_glsl_blinn_phong(in float NdotH, in float n)
{
    float nom = n*(n + 6.0) + 8.0;
    float denom = pow(0.5, 0.5*n) + n;
    float normalization = 0.125 * irr_glsl_RECIPROCAL_PI * nom/denom;
    return normalization*pow(NdotH, n);
}

vec3 irr_glsl_blinn_phong_fresnel_dielectric_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in float n, in vec3 ior)
{
    float denom = 4.0*inter.NdotV;
    return irr_glsl_blinn_phong(params.NdotH, n) * irr_glsl_fresnel_dielectric(ior, params.VdotH) / denom;
}

vec3 irr_glsl_blinn_phong_fresnel_conductor_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in float n, in mat2x3 ior)
{
    float denom = 4.0*inter.NdotV;
    return irr_glsl_blinn_phong(params.NdotH, n) * irr_glsl_fresnel_conductor(ior[0], ior[1], params.VdotH) / denom;
}

#endif
