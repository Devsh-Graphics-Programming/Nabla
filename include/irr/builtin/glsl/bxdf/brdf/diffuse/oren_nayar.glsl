#ifndef _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_
#define _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_

#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>

float irr_glsl_oren_nayar_cos_rec_pi_factored_out_wo_clamps(in float _a2, in float VdotL, in float maxNdotL, in float maxNdotV)
{
    // theta - polar angles
    // phi - azimuth angles
    float a2 = _a2*0.5; //todo read about this
    vec2 AB = vec2(1.0, 0.0) + vec2(-0.5, 0.45) * vec2(a2, a2)/vec2(a2+0.33, a2+0.09);
    float C = 1.0 / max(maxNdotL, maxNdotV);

    // should be equal to cos(phi)*sin(theta_i)*sin(theta_o)
    // where `phi` is the angle in the tangent plane to N, between L and V
    // and `theta_i` is the sine of the angle between L and N, similarily for `theta_o` but with V
    float cos_phi_sin_theta = max(VdotL-maxNdotL*maxNdotV,0.0);
    
    return (AB.x + AB.y * cos_phi_sin_theta * C);
}


float irr_glsl_oren_nayar_cos_eval_wo_clamps(in float a2, in float VdotL, in float maxNdotL, in float maxNdotV)
{
    return maxNdotL*irr_glsl_RECIPROCAL_PI*irr_glsl_oren_nayar_cos_rec_pi_factored_out_wo_clamps(a2,VdotL,maxNdotL,maxNdotV);
}

float irr_glsl_oren_nayar_cos_eval(in irr_glsl_LightSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction inter, in float a2)
{
    return irr_glsl_oren_nayar_cos_eval_wo_clamps(a2, _sample.VdotL, max(_sample.NdotL,0.0), max(inter.NdotV,0.0));
}


irr_glsl_LightSample irr_glsl_oren_nayar_cos_generate_wo_clamps(in vec3 tangentSpaceV, in mat3 m, in vec2 u)
{
    // until we find something better
    return irr_glsl_lambertian_cos_generate_wo_clamps(tangentSpaceV, m, u);
}
irr_glsl_LightSample irr_glsl_oren_nayar_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u, in float a2)
{
    return irr_glsl_oren_nayar_cos_generate_wo_clamps(irr_glsl_getTangentSpaceV(interaction),irr_glsl_getTangentFrame(interaction),u);
}


float irr_glsl_oren_nayar_pdf_wo_clamps(in float maxNdotL)
{
    return irr_glsl_lambertian_pdf_wo_clamps(maxNdotL);
}

float irr_glsl_oren_nayar_pdf(in irr_glsl_BxDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction i)
{
    return irr_glsl_lambertian_pdf(s, i);
}


float irr_glsl_oren_nayar_cos_remainder_and_pdf_wo_clamps(out float pdf, in float a2, in float VdotL, in float maxNdotL, in float maxNdotV)
{
    pdf = irr_glsl_oren_nayar_pdf_wo_clamps(maxNdotL);
    return irr_glsl_oren_nayar_cos_rec_pi_factored_out_wo_clamps(a2,VdotL,maxNdotL,maxNdotV);
}

float irr_glsl_oren_nayar_cos_remainder_and_pdf(out float pdf, in irr_glsl_LightSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in float a2)
{
    return irr_glsl_oren_nayar_cos_remainder_and_pdf_wo_clamps(pdf,a2,dot(interaction.V.dir,s.L), max(s.NdotL,0.0), max(interaction.NdotV,0.0));
}

#endif
