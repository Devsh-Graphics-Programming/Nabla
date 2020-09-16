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

irr_glsl_BxDFSample irr_glsl_lambertian_cos_generate_wo_clamps(in vec3 tangentSpaceV, in mat3 m, in vec2 u)
{
    vec3 L = irr_glsl_projected_hemisphere_generate(u);

    return irr_glsl_createBRDFSampleForDiffuse(tangentSpaceV,L,dot(tangentSpaceV,L),m);
}
irr_glsl_BxDFSample irr_glsl_lambertian_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u)
{
    const vec3 tangentSpaceV = vec3(interaction.TdotV,interaction.BdotV,interaction.isotropic.NdotV);
    return irr_glsl_lambertian_cos_generate_wo_clamps(tangentSpaceV,irr_glsl_getTangentFrame(interaction),u);
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
float irr_glsl_lambertian_cos_remainder_and_pdf(out float pdf, in irr_glsl_BxDFSample s)
{
    return irr_glsl_projected_hemisphere_remainder_and_pdf(pdf,max(s.NdotL,0.0));
}

#endif
