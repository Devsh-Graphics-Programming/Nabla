#ifndef _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_GGX_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_GGX_INCLUDED_

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/ndf/ggx.glsl>
#include <irr/builtin/glsl/bxdf/geom/smith/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>


vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float VdotH, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    float scalar_part = irr_glsl_ggx_height_correlated_aniso_cos_eval_DG_wo_clamps(NdotH2,TdotH2,BdotH2,maxNdotL,NdotL2,TdotL2,BdotL2,maxNdotV,NdotV2,TdotV2,BdotV2,ax,ax2,ay,ay2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr*scalar_part;
}

vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction inter, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    const float NdotH2 = params.isotropic.NdotH * params.isotropic.NdotH;
    const float TdotH2 = params.TdotH * params.TdotH;
    const float BdotH2 = params.BdotH * params.BdotH;

    const float TdotL2 = params.TdotL * params.TdotL;
    const float BdotL2 = params.BdotL * params.BdotL;

    const float TdotV2 = inter.TdotV * inter.TdotV;
    const float BdotV2 = inter.BdotV * inter.BdotV;
    return irr_glsl_ggx_height_correlated_aniso_cos_eval_wo_clamps(NdotH2, TdotH2, BdotH2, max(params.isotropic.NdotL,0.0), params.isotropic.NdotL_squared, TdotL2, BdotL2, max(inter.isotropic.NdotV, 0.0), inter.isotropic.NdotV_squared, TdotV2, BdotV2, params.isotropic.VdotH, ior, ax,ax2,ay,ay2);
}


vec3 irr_glsl_ggx_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    float scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG_wo_clamps(NdotH2, maxNdotL, NdotL2, maxNdotV, NdotV2, a2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr*scalar_part;
}

vec3 irr_glsl_ggx_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in mat2x3 ior, in float a2)
{
    const float NdotH2 = params.NdotH * params.NdotH;

    return irr_glsl_ggx_height_correlated_cos_eval_wo_clamps(NdotH2,max(params.NdotL,0.0),params.NdotL_squared, max(inter.NdotV,0.0), inter.NdotV_squared, params.VdotH,ior,a2);
}



irr_glsl_BxDFSample irr_glsl_ggx_transmitter_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 _sample, in float _ax, in float _ay)
{
    irr_glsl_BSDFSample _sample = irr_glsl_ggx_cos_generate(interaction,_sample.xy,_ax,_ay);
    return _sample;
}



float irr_glsl_ggx_pdf_wo_clamps(in float ndf, in float devsh_v, in float maxNdotV)
{
    return ndf * irr_glsl_GGXSmith_G1_wo_numerator(maxNdotV, devsh_v) * 0.5;
}
float irr_glsl_ggx_pdf_wo_clamps(in float NdotH2, in float maxNdotV, in float NdotV2, in float a2)
{
    float ndf = irr_glsl_ggx_trowbridge_reitz(a2, NdotH2);
    float devsh_v = irr_glsl_smith_ggx_devsh_part(NdotV2, a2, 1.0-a2);

    return irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);
}
float irr_glsl_ggx_pdf(in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction inter, in float a2)
{
    return irr_glsl_ggx_pdf_wo_clamps(s.NdotH*s.NdotH, max(inter.NdotV,0.0), inter.NdotV_squared, a2);
}

float irr_glsl_ggx_pdf_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float ax, in float ay, in float ax2, in float ay2)
{
    float ndf = irr_glsl_ggx_aniso(TdotH2,BdotH2,NdotH2, ax, ay, ax2, ay2);
    float devsh_v = irr_glsl_smith_ggx_devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);

    return irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);
}
float irr_glsl_ggx_pdf(in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in float ax, in float ay, in float ax2, in float ay2)
{
    float NdotH2 = s.NdotH * s.NdotH;
    float TdotH2 = s.TdotH * s.TdotH;
    float BdotH2 = s.BdotH * s.BdotH;

    float TdotV2 = interaction.TdotV * interaction.TdotV;
    float BdotV2 = interaction.BdotV * interaction.BdotV;

    return irr_glsl_ggx_pdf_wo_clamps(NdotH2,TdotH2,BdotH2, max(interaction.isotropic.NdotV,0.0),interaction.isotropic.NdotV_squared,TdotV2,BdotV2, ax,ay,ax2,ay2);
}



vec3 irr_glsl_ggx_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    float one_minus_a2 = 1.0 - a2;
    float devsh_v = irr_glsl_smith_ggx_devsh_part(NdotV2, a2, one_minus_a2);
    pdf = irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);

    float G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1_devsh(maxNdotL, NdotL2, maxNdotV, devsh_v, a2, one_minus_a2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr * G2_over_G1;
}

vec3 irr_glsl_ggx_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float a2)
{
    const float NdotH2 = s.NdotH * s.NdotH;

    const float NdotL2 = s.NdotL * s.NdotL;
    
    const float ndf = irr_glsl_ggx_trowbridge_reitz(a2, NdotH2);

    return irr_glsl_ggx_cos_remainder_and_pdf_wo_clamps(pdf, ndf, max(s.NdotL,0.0), NdotL2, max(interaction.NdotV,0.0), interaction.NdotV_squared, s.VdotH, ior, a2);
}


vec3 irr_glsl_ggx_aniso_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float VdotH, in mat2x3 ior, in float ax2,in float ay2)
{
    float devsh_v = irr_glsl_smith_ggx_devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);
    pdf = irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);

    float G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1(
        maxNdotL, TdotL2,BdotL2,NdotL2,
        maxNdotV, devsh_v,
        ax2, ay2
    );

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr * G2_over_G1;
}

vec3 irr_glsl_ggx_aniso_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float ax, in float ay)
{
    const float NdotH2 = s.NdotH * s.NdotH;
    const float TdotH2 = s.TdotH * s.TdotH;
    const float BdotH2 = s.BdotH * s.BdotH;

    const float NdotL2 = s.NdotL * s.NdotL;
    const float TdotL2 = s.TdotL * s.TdotL;
    const float BdotL2 = s.BdotL * s.BdotL;

    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;

    float ax2 = ax*ax;
    float ay2 = ay*ay;
    float ndf = irr_glsl_ggx_aniso(TdotH2,BdotH2,NdotH2, ax, ay, ax2, ay2);

	return irr_glsl_ggx_aniso_cos_remainder_and_pdf_wo_clamps(pdf, ndf, max(s.NdotL, 0.0), NdotL2, TdotL2, BdotL2, max(interaction.isotropic.NdotV, 0.0), TdotV2, BdotV2, interaction.isotropic.NdotV_squared, s.VdotH, ior, ax2, ay2);
}

#endif
