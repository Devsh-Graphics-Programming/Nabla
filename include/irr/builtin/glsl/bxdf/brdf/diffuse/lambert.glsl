#ifndef _IRR_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

#include <irr/builtin/glsl/bxdf/cos_weighted_sample.glsl>

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
    vec3 L = irr_glsl_projected_hemisphere_generate(u);

    irr_glsl_BSDFSample s;
    s.L = irr_glsl_getTangentFrame(interaction) * L;
    s.TdotL = L.x;
    s.BdotL = L.y;
    s.NdotL = L.z;
    /* Undefined
    s.TdotH = H.x;
    s.BdotH = H.y;
    s.NdotH = H.z;
    s.VdotH = VdotH;*/

    return s;
}

float irr_glsl_lambertian_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction)
{
    return irr_glsl_projected_hemisphere_remainder_and_pdf(pdf,max(s.NdotL,0.0));
}

#endif
