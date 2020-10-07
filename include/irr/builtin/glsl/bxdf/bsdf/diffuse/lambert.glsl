#ifndef _IRR_BXDF_BSDF_DIFFUSE_LAMBERT_INCLUDED_
#define _IRR_BXDF_BSDF_DIFFUSE_LAMBERT_INCLUDED_

#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/sampling/cos_weighted.glsl>

float irr_glsl_lambertian_transmitter()
{
    return irr_glsl_RECIPROCAL_PI*0.5;
}

float irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out_wo_clamps(in float absNdotL)
{
   return absNdotL;
}
float irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out(in float NdotL)
{
   return irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out_wo_clamps(abs(NdotL));
}

float irr_glsl_lambertian_transmitter_cos_eval_wo_clamps(in float absNdotL)
{
   return irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out_wo_clamps(absNdotL)*irr_glsl_lambertian_transmitter();
}
float irr_glsl_lambertian_transmitter_cos_eval(in irr_glsl_LightSample _sample)
{
   return irr_glsl_lambertian_transmitter_cos_eval_rec_2pi_factored_out(_sample.NdotL)*irr_glsl_lambertian_transmitter();
}

irr_glsl_LightSample irr_glsl_lambertian_transmitter_cos_generate_wo_clamps(in vec3 tangentSpaceV, in mat3 m, in vec3 u)
{
    vec3 L = irr_glsl_projected_sphere_generate(u);
    
    return irr_glsl_createLightSampleTangentSpace(tangentSpaceV,L,m);
}
irr_glsl_LightSample irr_glsl_lambertian_transmitter_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 u)
{
    return irr_glsl_lambertian_transmitter_cos_generate_wo_clamps(irr_glsl_getTangentSpaceV(interaction),irr_glsl_getTangentFrame(interaction),u);
}



float irr_glsl_lambertian_transmitter_pdf_wo_clamps(in float absNdotL)
{
    float pdf;
    irr_glsl_projected_sphere_remainder_and_pdf(pdf, absNdotL);
    return pdf;
}



float irr_glsl_lambertian_transmitter_cos_remainder_and_pdf_wo_clamps(out float pdf, in float absNdotL)
{
    return irr_glsl_projected_sphere_remainder_and_pdf(pdf,absNdotL);
}
float irr_glsl_lambertian_transmitter_cos_remainder_and_pdf(out float pdf, in irr_glsl_LightSample s)
{
    return irr_glsl_lambertian_transmitter_cos_remainder_and_pdf_wo_clamps(pdf,abs(s.NdotL));
}

#endif
