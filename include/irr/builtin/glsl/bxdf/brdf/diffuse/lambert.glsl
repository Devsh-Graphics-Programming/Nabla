#ifndef _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BSDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

#include <irr/builtin/glsl/bxdf/common_samples.glsl>

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
    vec3 L;
    L.xy = irr_glsl_concentricMapping(u);
    L.z = sqrt(1.0-dot(L.xy,L.xy));

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

float irr_glsl_lambertian_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction)
{
    pdf = irr_glsl_lambertian()*max(s.NdotL,0.0);
    return 1.0;
}

#endif
