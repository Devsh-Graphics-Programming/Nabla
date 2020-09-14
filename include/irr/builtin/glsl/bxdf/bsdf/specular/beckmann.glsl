#ifndef _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_BECKMANN_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_BSDF_SPECULAR_BECKMANN_INCLUDED_

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/ndf/beckmann.glsl>
#include <irr/builtin/glsl/bxdf/geom/smith/beckmann.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/beckmann.glsl>

#include <irr/builtin/glsl/math/functions.glsl>

irr_glsl_BxDFSample irr_glsl_beckmann_transmitter_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample, in float ax, in float ay)
{
    irr_glsl_BxDFSample _sample = irr_glsl_beckmann_cos_generate(interaction, _sample.xy, _ax, _ay); // TODO
    return _sample;
}



float irr_glsl_beckmann_transmitter_pdf_wo_clamps(in float ndf, in float lambdaV, in float maxNdotV)
{
    return irr_glsl_beckmann_pdf_wo_clamps(ndf, lambda, maxNdotV);
}

float irr_glsl_beckmann_transmitter_pdf_wo_clamps(in float NdotH2, in float maxNdotV, in float NdotV2, in float a2)
{
    return irr_glsl_beckmann_pdf_wo_clamps(NdotH2, maxNdotV, maxNdotV, a2);
}

float irr_glsl_beckmann_transmitter_pdf_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float ax, in float ax2, in float ay, in float ay2)
{
    return irr_glsl_beckmann_pdf_wo_clamps(NdotH2,TdotH2,BdotH2,maxNdotV,TdotV2,BdotV2,NdotV2,ax,ax2,ay,ay2);
}


// TODO
vec3 irr_glsl_beckmann_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    float lambda_V = irr_glsl_smith_beckmann_Lambda(NdotV2, a2);
    float onePlusLambda_V = 1.0 + lambda_V;

    float G1 = 1.0 / onePlusLambda_V;
    pdf = ndf * G1 * 0.25 / maxNdotV;

    float G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(onePlusLambda_V, maxNdotL, NdotL2, a2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr * G2_over_G1;
}
vec3 irr_glsl_beckmann_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float a2)
{
    const float NdotH2 = s.NdotH * s.NdotH;

    const float NdotL2 = s.NdotL * s.NdotL;

    const float ndf = irr_glsl_beckmann(a2, NdotH2);
	
    return irr_glsl_beckmann_cos_remainder_and_pdf_wo_clamps(pdf, ndf, max(s.NdotL,0.0), NdotL2, max(interaction.NdotV,0.0), interaction.NdotV_squared, s.VdotH, ior, a2);
}


// TODO
vec3 irr_glsl_beckmann_aniso_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float VdotH, in mat2x3 ior, in float ax2, in float ay2)
{
    float c2 = irr_glsl_smith_beckmann_C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
    float lambda_V = irr_glsl_smith_beckmann_Lambda(c2);
    float onePlusLambda_V = 1.0 + lambda_V;

    float G1 = 1.0 / onePlusLambda_V;
    pdf = ndf * G1 * 0.25 / maxNdotV;

    float G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(onePlusLambda_V, maxNdotL, TdotL2, BdotL2, NdotL2, ax2, ay2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr * G2_over_G1;
}
vec3 irr_glsl_beckmann_aniso_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    const float NdotH2 = s.NdotH * s.NdotH;
    const float TdotH2 = s.TdotH * s.TdotH;
    const float BdotH2 = s.BdotH * s.BdotH;

    const float NdotL2 = s.NdotL * s.NdotL;
    const float TdotL2 = s.TdotL * s.TdotL;
    const float BdotL2 = s.BdotL * s.BdotL;
    
    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;
    
    const float ndf = irr_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);

	return irr_glsl_beckmann_aniso_cos_remainder_and_pdf_wo_clamps(pdf, ndf, max(s.NdotL,0.0), NdotL2,TdotL2,BdotL2, max(interaction.isotropic.NdotV,0.0),TdotV2,BdotV2, interaction.isotropic.NdotV_squared, s.VdotH, ior, ax2, ay2);
}



vec3 irr_glsl_beckmann_smith_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    float scalar_part = irr_glsl_beckmann_smith_height_correlated_cos_eval_DG_wo_clamps(NdotH2, NdotL2, maxNdotV, NdotV2, a2);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    
    return scalar_part*fr;
}
vec3 irr_glsl_beckmann_smith_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float a2)
{
    const float NdotH2 = params.NdotH * params.NdotH;

    return irr_glsl_beckmann_smith_height_correlated_cos_eval_wo_clamps(NdotH2,params.NdotL_squared,max(interaction.NdotV,0.0),interaction.NdotV_squared,params.VdotH,ior,a2);
}


vec3 irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float VdotH, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    float scalar_part = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_DG_wo_clamps(NdotH2,TdotH2,BdotH2, NdotL2,TdotL2,BdotL2, maxNdotV,NdotV2,TdotV2,BdotV2, ax, ax2, ay, ay2);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    
    return scalar_part*fr;
}
vec3 irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    const float NdotH2 = params.isotropic.NdotH * params.isotropic.NdotH;
    const float TdotH2 = params.TdotH * params.TdotH;
    const float BdotH2 = params.BdotH * params.BdotH;

    const float TdotL2 = params.TdotL * params.TdotL;
    const float BdotL2 = params.BdotL * params.BdotL;

    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;

    return irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_wo_clamps(NdotH2,TdotH2,BdotH2, params.isotropic.NdotL_squared,TdotL2,BdotL2, max(interaction.isotropic.NdotV,0.0),interaction.isotropic.NdotV_squared,TdotV2,BdotV2, params.isotropic.VdotH, ior,ax,ax2,ay,ay2);
}

#endif
