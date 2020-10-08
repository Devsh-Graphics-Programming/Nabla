#ifndef _IRR_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BXDF_BRDF_DIFFUSE_LAMBERT_INCLUDED_

#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/sampling/cos_weighted.glsl>

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
float irr_glsl_lambertian_cos_eval(in irr_glsl_LightSample _sample)
{
    return irr_glsl_lambertian_cos_eval_rec_pi_factored_out(_sample.NdotL)*irr_glsl_lambertian();
}

irr_glsl_LightSample irr_glsl_lambertian_cos_generate_wo_clamps(in vec3 tangentSpaceV, in mat3 m, in vec2 u)
{
    vec3 L = irr_glsl_projected_hemisphere_generate(u);

    return irr_glsl_createLightSampleTangentSpace(tangentSpaceV,L,m);
}
irr_glsl_LightSample irr_glsl_lambertian_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u)
{
    return irr_glsl_lambertian_cos_generate_wo_clamps(irr_glsl_getTangentSpaceV(interaction),irr_glsl_getTangentFrame(interaction),u);
}



float irr_glsl_lambertian_pdf_wo_clamps(in float maxNdotL)
{
    float pdf;
    irr_glsl_projected_hemisphere_remainder_and_pdf(pdf,maxNdotL);
    return pdf;
}



float irr_glsl_lambertian_cos_remainder_and_pdf_wo_clamps(out float pdf, in float maxNdotL)
{
    return irr_glsl_projected_hemisphere_remainder_and_pdf(pdf,maxNdotL);
}
float irr_glsl_lambertian_cos_remainder_and_pdf(out float pdf, in irr_glsl_LightSample s)
{
    return irr_glsl_projected_hemisphere_remainder_and_pdf(pdf,max(s.NdotL,0.0));
}

#endif
