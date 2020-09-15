#ifndef _IRR_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

#include <irr/builtin/glsl/bxdf/cos_weighted_sample.glsl>

float irr_glsl_lambertian()
{
    return irr_glsl_RECIPROCAL_PI;
}

float irr_glsl_lambertian_cos_eval_rec_pi_factored_out_wo_clamps(in float maxNdotL)
{
   return maxNdotL;
}
float irr_glsl_lambertian_cos_eval_rec_pi_factored_out(in float NdotL)
{
   return irr_glsl_lambertian_cos_eval_rec_pi_factored_out_wo_clamps(max(NdotL,0.0));
}

float irr_glsl_lambertian_cos_eval_wo_clamps(in float maxNdotL)
{
   return irr_glsl_lambertian_cos_eval_rec_pi_factored_out_wo_clamps(maxNdotL)*irr_glsl_lambertian();
}
float irr_glsl_lambertian_cos_eval(in irr_glsl_BSDFIsotropicParams params)
{
    return irr_glsl_lambertian_cos_eval_rec_pi_factored_out(params.NdotL)*irr_glsl_lambertian();
}

irr_glsl_BxDFSample irr_glsl_lambertian_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u)
{
    vec3 L = irr_glsl_projected_hemisphere_generate(u);

    irr_glsl_BxDFSample s;
    s.L = irr_glsl_getTangentFrame(interaction)*L;
    s.TdotL = L.x;
    s.BdotL = L.y;
    s.NdotL = L.z;
    //assuming H=N, i have to set them to something to jump between diffuse and specular generators (plastic)
    s.TdotH = 0.0;
    s.BdotH = 0.0;
    s.NdotH = 1.0;
    s.VdotH = L.z;

    return s;
}


float irr_glsl_lambertian_pdf_wo_clamps(in float maxNdotL)
{
    return irr_glsl_projected_hemisphere_pdf(maxNdotL);
}


float irr_glsl_lambertian_cos_remainder_and_pdf_wo_clamps(out float pdf, in float maxNdotL)
{
    return irr_glsl_projected_hemisphere_remainder_and_pdf(pdf,maxNdotL);
}
float irr_glsl_lambertian_cos_remainder_and_pdf(out float pdf, in irr_glsl_BxDFSample s)
{
    return irr_glsl_projected_hemisphere_remainder_and_pdf(pdf,max(s.NdotL,0.0));
}

#endif
