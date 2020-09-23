#ifndef _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_GGX_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_GGX_INCLUDED_

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/ndf/ggx.glsl>
#include <irr/builtin/glsl/bxdf/geom/smith/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>


vec3 irr_glsl_ggx_transmitter_height_correlated_aniso_cos_eval_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float absNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float absNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float VdotH, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    float scalar_part = irr_glsl_ggx_height_correlated_aniso_cos_eval_DG_wo_clamps(NdotH2,TdotH2,BdotH2,absNdotL,NdotL2,TdotL2,BdotL2,absNdotV,NdotV2,TdotV2,BdotV2,ax,ax2,ay,ay2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr*scalar_part;
}

vec3 irr_glsl_ggx_transmitter_height_correlated_aniso_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction inter, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
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


vec3 irr_glsl_ggx_transmitter_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float absNdotL, in float NdotL2, in float absNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    float scalar_part = irr_glsl_ggx_height_correlated_cos_eval_DG_wo_clamps(NdotH2, absNdotL, NdotL2, absNdotV, NdotV2, a2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr*scalar_part;
}

vec3 irr_glsl_ggx_transmitter_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in mat2x3 ior, in float a2)
{
    const float NdotH2 = params.NdotH * params.NdotH;

    return irr_glsl_ggx_height_correlated_cos_eval_wo_clamps(NdotH2,max(params.NdotL,0.0),params.NdotL_squared, max(inter.NdotV,0.0), inter.NdotV_squared, params.VdotH,ior,a2);
}



irr_glsl_BxDFSample irr_glsl_ggx_transmitter_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec3 _sample, in float _ax, in float _ay)
{
    irr_glsl_BSDFSample _sample = irr_glsl_ggx_cos_generate(interaction,_sample.xy,_ax,_ay);
    return _sample;
}



float irr_glsl_ggx_transmitter_pdf_wo_clamps(in float ndf, in float reflectance, in float devsh_v, in float absNdotV)
{
    return ndf * reflectance * irr_glsl_GGXSmith_G1_wo_numerator(absNdotV, devsh_v) * 0.5;
}
float irr_glsl_ggx_transmitter_pdf_wo_clamps(in float NdotH2, in float absNdotV, in float NdotV2, in float absVdotH, in float orientedEta2, in float a2)
{
    const float ndf = irr_glsl_ggx_trowbridge_reitz(a2, NdotH2);
    const float reflectance = irr_glsl_fresnel_dielectric_common(orientedEta2,absVdotH);
    const float devsh_v = irr_glsl_smith_ggx_devsh_part(NdotV2, a2, 1.0 - a2);

    return irr_glsl_ggx_pdf_wo_clamps(ndf, reflectance, devsh_v, absNdotV);
}

float irr_glsl_ggx_transmitter_pdf_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float absNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float absVdotH, in float orientedEta2, in float ax, in float ay, in float ax2, in float ay2)
{
    const float ndf = irr_glsl_ggx_aniso(TdotH2, BdotH2, NdotH2, ax, ay, ax2, ay2);
    const float reflectance = irr_glsl_fresnel_dielectric_common(orientedEta2, absVdotH);
    const float devsh_v = irr_glsl_smith_ggx_devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);

    return irr_glsl_ggx_pdf_wo_clamps(ndf, reflectance, devsh_v, absNdotV);
}



vec3 irr_glsl_ggx_transmitter_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float absNdotL, in float NdotL2, in float absNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    float one_minus_a2 = 1.0 - a2;
    float devsh_v = irr_glsl_smith_ggx_devsh_part(NdotV2, a2, one_minus_a2);
    pdf = irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, absNdotV);

    float G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1_devsh(absNdotL, NdotL2, absNdotV, devsh_v, a2, one_minus_a2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr * G2_over_G1;
}

vec3 irr_glsl_ggx_transmitter_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float a2)
{
    const float NdotH2 = s.NdotH * s.NdotH;

    const float NdotL2 = s.NdotL * s.NdotL;
    
    const float ndf = irr_glsl_ggx_trowbridge_reitz(a2, NdotH2);

    return irr_glsl_ggx_cos_remainder_and_pdf_wo_clamps(pdf, ndf, max(s.NdotL,0.0), NdotL2, max(interaction.NdotV,0.0), interaction.NdotV_squared, s.VdotH, ior, a2);
}


vec3 irr_glsl_ggx_transmitter_aniso_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float absNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float absNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float VdotH, in mat2x3 ior, in float ax2,in float ay2)
{
    float devsh_v = irr_glsl_smith_ggx_devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);
    pdf = irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, absNdotV);

    float G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1(
        absNdotL, TdotL2,BdotL2,NdotL2,
        absNdotV, devsh_v,
        ax2, ay2
    );

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr * G2_over_G1;
}

float irr_glsl_ggx_transmitter_aniso_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float ax, in float ay)
{
    const float TdotH2 = _cache.TdotH*_cache.TdotH;
    const float BdotH2 = _cache.BdotH*_cache.BdotH;

    const float TdotL2 = _sample.TdotL*_sample.TdotL;
    const float BdotL2 = _sample.BdotL*_sample.BdotL;

    const float TdotV2 = interaction.TdotV*interaction.TdotV;
    const float BdotV2 = interaction.BdotV*interaction.BdotV;

    const float ax2 = ax*ax;
    const float ay2 = ay*ay;
    const float ndf = irr_glsl_ggx_aniso(TdotH2,BdotH2,_cache.isotropic.NdotH2, ax, ay, ax2, ay2);
    const float reflectance = irr_glsl_fresnel_dielectric_common(orientedEta2, absVdotH);

	return irr_glsl_ggx_transmitter_aniso_cos_remainder_and_pdf_wo_clamps(pdf, ndf, abs(_sample.NdotL), _sample.NdotL2, TdotL2, BdotL2, abs(interaction.isotropic.NdotV), TdotV2, BdotV2, interaction.isotropic.NdotV_squared, _cache.isotropic.VdotH, ior, ax2, ay2);
}

#endif
