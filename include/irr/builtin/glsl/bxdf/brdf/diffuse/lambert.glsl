#ifndef _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

#include <irr/builtin/glsl/bxdf/brdf/cos_weighted_sample.glsl>

float irr_glsl_lambertian()
{
    return irr_glsl_RECIPROCAL_PI;
}

float irr_glsl_lambertian_cos_eval_rec_pi_factored_out(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter)
{
   return max(params.NdotL,0.0);
}

float irr_glsl_lambertian_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter)
{
   return irr_glsl_lambertian_cos_eval_rec_pi_factored_out(params,inter)*irr_glsl_lambertian();
}

irr_glsl_BSDFSample irr_glsl_lambertian_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u)
{
    return irr_glsl_cos_weighted_cos_generate(interaction,u);
}

float irr_glsl_lambertian_pdf(in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction)
{
    return irr_glsl_cos_weighted_pdf(s, interaction);
}

vec3 irr_glsl_lambertian_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction)
{
    return irr_glsl_cos_weighted_cos_remainder_and_pdf(pdf,s,interaction);
}

#endif
