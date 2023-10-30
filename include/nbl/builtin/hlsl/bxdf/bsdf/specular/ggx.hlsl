// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_BSDF_SPECULAR_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BSDF_SPECULAR_GGX_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/transmission.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf/ggx.hlsl>
#include <nbl/builtin/hlsl/bxdf/geom/smith/ggx.hlsl>
#include <nbl/builtin/hlsl/bxdf/brdf/specular/ggx.hlsl>


namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace bsdf
{
namespace specular
{


float ggx_height_correlated_aniso_cos_eval_wo_clamps(
    in float NdotH2, in float TdotH2, in float BdotH2,
    in float absNdotL, in float NdotL2, in float TdotL2, in float BdotL2,
    in float absNdotV, in float NdotV2, in float TdotV2, in float BdotV2,
    in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH,
    in float orientedEta, in float orientedEta2,
    in float ax, in float ax2, in float ay, in float ay2)
{
    float NG_already_in_reflective_dL_measure = brdf::specular::ggx_height_correlated_aniso_cos_eval_DG_wo_clamps(NdotH2, TdotH2, BdotH2, absNdotL, NdotL2, TdotL2, BdotL2, absNdotV, NdotV2, TdotV2, BdotV2, ax, ax2, ay, ay2);

    const float reflectance = fresnel::dielectric_common(orientedEta2, abs(VdotH));

    return reflectance * ndf::ggx::microfacet_to_light_measure_transform(NG_already_in_reflective_dL_measure, absNdotL, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
}

// before calling you must ensure that `AnisotropicMicrofacetCache` is valid (if a given V vector can "see" the L vector)
template <class IncomingRayDirInfo>
float ggx_height_correlated_aniso_cos_eval(in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, in AnisotropicMicrofacetCache _cache, in float eta, in float ax, in float ay)
{
    const float TdotH2 = _cache.TdotH * _cache.TdotH;
    const float BdotH2 = _cache.BdotH * _cache.BdotH;

    const float TdotL2 = _sample.TdotL * _sample.TdotL;
    const float BdotL2 = _sample.BdotL * _sample.BdotL;

    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;

    const float VdotH = _cache.VdotH;

    float orientedEta, dummy;
    const bool backside = math::getOrientedEtas(orientedEta, dummy, VdotH, eta);
    const float orientedEta2 = orientedEta * orientedEta;

    const float VdotHLdotH = VdotH * _cache.LdotH;
    const bool transmitted = VdotHLdotH < 0.0;

    return ggx_height_correlated_aniso_cos_eval_wo_clamps(
        _cache.NdotH2, TdotH2, BdotH2,
        abs(_sample.NdotL), _sample.NdotL2, TdotL2, BdotL2,
        abs(interaction.NdotV), interaction.NdotV_squared, TdotV2, BdotV2,
        transmitted, VdotH, _cache.LdotH, VdotHLdotH, orientedEta, orientedEta2,
        ax, ax * ax, ay, ay * ay
    );
}


float ggx_height_correlated_cos_eval_wo_clamps(
    in float NdotH2, in float absNdotL, in float NdotL2,
    in float absNdotV, in float NdotV2,
    in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH,
    in float orientedEta, in float orientedEta2, in float a2)
{
    const float NG_already_in_reflective_dL_measure = brdf::specular::ggx_height_correlated_cos_eval_DG_wo_clamps(NdotH2, absNdotL, NdotL2, absNdotV, NdotV2, a2);

    const float reflectance = fresnel::dielectric_common(orientedEta2, abs(VdotH));

    return reflectance * ndf::ggx::microfacet_to_light_measure_transform(NG_already_in_reflective_dL_measure, absNdotL, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
}

// before calling you must ensure that `AnisotropicMicrofacetCache` is valid (if a given V vector can "see" the L vector)
template <class IncomingRayDirInfo>
float ggx_height_correlated_cos_eval(in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Isotropic<IncomingRayDirInfo> interaction, in IsotropicMicrofacetCache _cache, in float eta, in float a2)
{
    float orientedEta, dummy;
    const bool backside = math::getOrientedEtas(orientedEta, dummy, _cache.VdotH, eta);
    const float orientedEta2 = orientedEta * orientedEta;

    const float VdotHLdotH = _cache.VdotH * _cache.LdotH;
    const bool transmitted = VdotHLdotH < 0.0;

    return ggx_height_correlated_cos_eval_wo_clamps(
        _cache.NdotH2, abs(_sample.NdotL), _sample.NdotL2,
        abs(interaction.NdotV), interaction.NdotV_squared,
        transmitted, _cache.VdotH, _cache.LdotH, VdotHLdotH, orientedEta, orientedEta2, a2
    );
}

// TODO: unifty the two following functions into `microfacet_BSDF_cos_generate_wo_clamps(float3 H,...)` and `microfacet_BSDF_cos_generate` or at least a auto declaration macro in lieu of a template
template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> ggx_cos_generate_wo_clamps(in float3 localV, in bool backside, in float3 upperHemisphereLocalV, in float3x3 m, inout float3 u, in float _ax, in float _ay, in float rcpOrientedEta, in float orientedEta2, in float rcpOrientedEta2, out AnisotropicMicrofacetCache _cache)
{
    // thanks to this manouvre the H will always be in the upper hemisphere (NdotH>0.0)
    const float3 H = brdf::specular::ggx_cos_generate(upperHemisphereLocalV, u.xy, _ax, _ay);

    const float VdotH = dot(localV, H);
    const float reflectance = fresnel::dielectric_common(orientedEta2, abs(VdotH));

    float rcpChoiceProb;
    bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

    float3 localL;
    _cache = AnisotropicMicrofacetCache::create(localV, H, transmitted, rcpOrientedEta, rcpOrientedEta2);
    localL = math::reflect_refract(transmitted, localV, H, VdotH, _cache.LdotH, rcpOrientedEta);

    return LightSample<IncomingRayDirInfo>::createTangentSpace(localV, IncomingRayDirInfo::create(localL), m);
}

template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> ggx_cos_generate(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, inout float3 u, in float ax, in float ay, in float eta, out AnisotropicMicrofacetCache _cache)
{
    const float3 localV = interaction.getTangentSpaceV();

    float orientedEta, rcpOrientedEta;
    const bool backside = math::getOrientedEtas(orientedEta, rcpOrientedEta, interaction.NdotV, eta);

    const float3 upperHemisphereV = backside ? (-localV) : localV;

    const float3x3 m = interaction.getTangentFrame();
    return ggx_cos_generate_wo_clamps<IncomingRayDirInfo>(localV, backside, upperHemisphereV, m, u, ax, ay, rcpOrientedEta, orientedEta * orientedEta, rcpOrientedEta * rcpOrientedEta, _cache);
}



float ggx_pdf_wo_clamps(in bool transmitted, in float reflectance, in float ndf, in float devsh_v, in float absNdotV, in float VdotH, in float LdotH, in float VdotHLdotH, in float orientedEta)
{
    return geom_smith::VNDF_pdf_wo_clamps(ndf, geom_smith::ggx::G1_wo_numerator(absNdotV, devsh_v), absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta, reflectance);
}
float ggx_pdf_wo_clamps(in bool transmitted, in float reflectance, in float NdotH2, in float absNdotV, in float NdotV2, in float VdotH, in float LdotH, in float VdotHLdotH, in float a2, in float orientedEta)
{
    const float ndf = ndf::ggx::trowbridge_reitz(a2, NdotH2);
    const float devsh_v = geom_smith::ggx::devsh_part(NdotV2, a2, 1.0 - a2);

    return ggx_pdf_wo_clamps(transmitted, reflectance, ndf, devsh_v, absNdotV, VdotH, LdotH, VdotHLdotH, orientedEta);
}

float ggx_pdf_wo_clamps(in bool transmitted, in float reflectance, in float NdotH2, in float TdotH2, in float BdotH2, in float absNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float VdotH, in float LdotH, in float VdotHLdotH, in float ax, in float ay, in float ax2, in float ay2, in float orientedEta)
{
    const float ndf = ndf::ggx::aniso(TdotH2, BdotH2, NdotH2, ax, ay, ax2, ay2);
    const float devsh_v = geom_smith::ggx::devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);

    return ggx_pdf_wo_clamps(transmitted, reflectance, ndf, devsh_v, absNdotV, VdotH, LdotH, VdotHLdotH, orientedEta);
}

float ggx_cos_quotient_and_pdf_wo_clamps(out float pdf, in float ndf, in bool transmitted, in float absNdotL, in float NdotL2, in float absNdotV, in float NdotV2, in float VdotH, in float LdotH, in float VdotHLdotH, in float reflectance, in float orientedEta, in float a2)
{
    const float one_minus_a2 = 1.0 - a2;
    const float devsh_v = geom_smith::ggx::devsh_part(NdotV2, a2, one_minus_a2);
    pdf = ggx_pdf_wo_clamps(transmitted, reflectance, ndf, devsh_v, absNdotV, VdotH, LdotH, VdotHLdotH, orientedEta);

    return geom_smith::ggx::G2_over_G1_devsh(absNdotL, NdotL2, absNdotV, devsh_v, a2, one_minus_a2);
}

template <class IncomingRayDirInfo>
float ggx_cos_quotient_and_pdf(out float pdf, in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Isotropic<IncomingRayDirInfo> interaction, in IsotropicMicrofacetCache _cache, in float eta, in float a2)
{
    const float ndf = ndf::ggx::trowbridge_reitz(a2, _cache.NdotH2);

    float orientedEta, dummy;
    const bool backside = math::getOrientedEtas(orientedEta, dummy, _cache.VdotH, eta);
    const float orientedEta2 = orientedEta * orientedEta;

    const float VdotHLdotH = _cache.VdotH * _cache.LdotH;
    const bool transmitted = VdotHLdotH < 0.0;

    const float reflectance = fresnel::dielectric_common(orientedEta2, abs(_cache.VdotH));

    const float absNdotV = abs(interaction.NdotV);
    return ggx_cos_quotient_and_pdf_wo_clamps(pdf, ndf, transmitted, abs(_sample.NdotL), _sample.NdotL2, absNdotV, interaction.NdotV_squared, _cache.VdotH, _cache.LdotH, VdotHLdotH, reflectance, orientedEta, a2);
}

template <class IncomingRayDirInfo>
float ggx_aniso_cos_quotient_and_pdf_wo_clamps(out float pdf, in float ndf, in bool transmitted, in float absNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float absNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float VdotH, in float LdotH, in float VdotHLdotH, in float reflectance, in float orientedEta, in float ax2, in float ay2)
{
    const float devsh_v = geom_smith::ggx::devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);
    pdf = ggx_pdf_wo_clamps(transmitted, reflectance, ndf, devsh_v, absNdotV, VdotH, LdotH, VdotHLdotH, orientedEta);

    return geom_smith::ggx::G2_over_G1_devsh(
        absNdotL, TdotL2, BdotL2, NdotL2,
        absNdotV, devsh_v,
        ax2, ay2
    );
}

template <class IncomingRayDirInfo>
float ggx_aniso_cos_quotient_and_pdf(out float pdf, in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, in AnisotropicMicrofacetCache _cache, in float eta, in float ax, in float ay)
{
    const float ax2 = ax * ax;
    const float ay2 = ay * ay;
    const float TdotH2 = _cache.TdotH * _cache.TdotH;
    const float BdotH2 = _cache.BdotH * _cache.BdotH;
    const float ndf = ndf::ggx::aniso(TdotH2, BdotH2, _cache.NdotH2, ax, ay, ax2, ay2);

    const float TdotL2 = _sample.TdotL * _sample.TdotL;
    const float BdotL2 = _sample.BdotL * _sample.BdotL;

    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;

    const float VdotH = _cache.VdotH;

    float orientedEta, dummy;
    const bool backside = math::getOrientedEtas(orientedEta, dummy, VdotH, eta);
    const float orientedEta2 = orientedEta * orientedEta;

    const float VdotHLdotH = VdotH * _cache.LdotH;
    const bool transmitted = VdotHLdotH < 0.0;

    const float reflectance = fresnel::dielectric_common(orientedEta2, abs(VdotH));

    const float absNdotV = abs(interaction.NdotV);
    return ggx_aniso_cos_quotient_and_pdf_wo_clamps(pdf, ndf, transmitted, abs(_sample.NdotL), _sample.NdotL2, TdotL2, BdotL2, absNdotV, TdotV2, BdotV2, interaction.NdotV_squared, VdotH, _cache.LdotH, VdotHLdotH, reflectance, orientedEta, ax2, ay2);
}

}
}
}
}
}

#endif
