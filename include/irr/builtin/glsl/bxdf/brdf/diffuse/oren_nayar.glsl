#ifndef _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_
#define _BRDF_DIFFUSE_OREN_NAYAR_INCLUDED_

#include <irr/builtin/glsl/bxdf/common.glsl>

float irr_glsl_oren_nayar(in float _a2, in float VdotL, in float NdotL, in float NdotV)
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
    
    return (AB.x + AB.y * cos_phi_sin_theta * C) * irr_glsl_RECIPROCAL_PI;
}

float irr_glsl_oren_nayar_cos_eval(in irr_glsl_BSDFIsotropicParams params, in float a2)
{
    return params.NdotL * irr_glsl_oren_nayar(a2, params.VdotL, params.NdotL, params.interaction.NdotV);
}

#endif
