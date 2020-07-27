#ifndef _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_
#define _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_

#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>

float irr_glsl_oren_nayar_rec_pi_factored_out(in float _a2, in float VdotL, in float NdotL, in float NdotV)
{
    // theta - polar angles
    // phi - azimuth angles
    float a2 = _a2*0.5; //todo read about this
    vec2 AB = vec2(1.0, 0.0) + vec2(-0.5, 0.45) * vec2(a2, a2)/vec2(a2+0.33, a2+0.09);
    float C = 1.0 / max(NdotL, NdotV);

    // should be equal to cos(phi)*sin(theta_i)*sin(theta_o)
    // where `phi` is the angle in the tangent plane to N, between L and V
    // and `theta_i` is the sine of the angle between L and N, similarily for `theta_o` but with V
    float cos_phi_sin_theta = max(VdotL-NdotL*NdotV,0.0);
    
    return (AB.x + AB.y * cos_phi_sin_theta * C);
}

float irr_glsl_oren_nayar_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in float a2)
{
    return max(params.NdotL,0.0)* irr_glsl_RECIPROCAL_PI * irr_glsl_oren_nayar_rec_pi_factored_out(a2, params.VdotL, params.NdotL, inter.NdotV);
}


irr_glsl_BSDFSample irr_glsl_oren_nayar_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u, in float a2)
{
    // until we find something better
    return irr_glsl_lambertian_cos_generate(interaction,u);
}

float irr_glsl_oren_nayar_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in float a2)
{
    // paired with the generation function
    irr_glsl_lambertian_cos_remainder_and_pdf(pdf,s,interaction);
    // but the remainder is different
    return irr_glsl_oren_nayar_rec_pi_factored_out(a2, dot(interaction.isotropic.V.dir,s.L), s.NdotL, interaction.isotropic.NdotV);
}

#endif
