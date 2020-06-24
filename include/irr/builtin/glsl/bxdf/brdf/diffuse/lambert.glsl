#ifndef _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

#include <irr/builtin/glsl/bxdf/common.glsl>

float irr_glsl_lambertian()
{
    return irr_glsl_RECIPROCAL_PI;
}

float irr_glsl_lambertian_cos_eval_rec_pi_factored_out(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter)
{
   return params.NdotL;
}

float irr_glsl_lambertian_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter)
{
   return irr_glsl_lambertian_cos_eval_rec_pi_factored_out(params,inter)*irr_glsl_lambertian();
}

#endif
