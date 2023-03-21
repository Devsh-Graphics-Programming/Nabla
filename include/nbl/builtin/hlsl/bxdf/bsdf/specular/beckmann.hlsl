// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_BSDF_SPECULAR_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BSDF_SPECULAR_BECKMANN_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/ndf/beckmann.hlsl>
#include <nbl/builtin/hlsl/bxdf/geom/smith/beckmann.hlsl>
#include <nbl/builtin/hlsl/bxdf/brdf/specular/beckmann.hlsl>
#include <nbl/builtin/hlsl/bxdf/bsdf/specular/common.hlsl>

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

// TODO why `backside` is here? its not used. is it some convention?
template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> beckmann_cos_generate_wo_clamps(in float3 localV, in bool backside, in float3 upperHemisphereLocalV, in float3x3 m, float3 u, in float ax, in float ay, in float rcpOrientedEta, in float orientedEta2, in float rcpOrientedEta2, out AnisotropicMicrofacetCache _cache)
{
    // thanks to this manouvre the H will always be in the upper hemisphere (NdotH>0.0)
    const float3 H = brdf::specular::beckmann_cos_generate_wo_clamps(upperHemisphereLocalV,u.xy,ax,ay);

    const float VdotH = dot(localV,H);
    const float reflectance = fresnel::dielectric_common(orientedEta2,abs(VdotH));
    
    float rcpChoiceProb;
    const bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);
    
    float3 localL;
    _cache = AnisotropicMicrofacetCache::create(localV, H, transmitted, rcpOrientedEta, rcpOrientedEta2);
    localL = math::reflect_refract(transmitted, localV, H, VdotH, _cache.LdotH, rcpOrientedEta);
    
    return LightSample<IncomingRayDirInfo>::createTangentSpace(localV, IncomingRayDirInfo::create(localL), m);
}

// `u` should be inout right?
template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> beckmann_cos_generate(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, inout float3 u, in float ax, in float ay, in float eta, out AnisotropicMicrofacetCache _cache)
{
    const float3 localV = interaction.getTangentSpaceV();
    
    float orientedEta, rcpOrientedEta;
    const bool backside = getOrientedEtas(orientedEta, rcpOrientedEta, interaction.NdotV, eta);
    
    const float3 upperHemisphereV = backside ? (-localV):localV;

    const float3x3 m = interaction.getTangentFrame();
    return beckmann_cos_generate_wo_clamps<IncomingRayDirInfo>(localV,backside,upperHemisphereV,m, u,ax,ay, rcpOrientedEta,orientedEta*orientedEta,rcpOrientedEta*rcpOrientedEta,_cache);
}



// isotropic PDF
float beckmann_pdf_wo_clamps(in bool transmitted, in float reflectance, in float ndf, in float absNdotV, in float NdotV2, in float VdotH, in float LdotH, in float VdotHLdotH, in float a2, in float orientedEta, out float onePlusLambda_V)
{
    const float lambda = geom_smith::beckmann::Lambda(NdotV2, a2);
    return geom_smith::VNDF_pdf_wo_clamps(ndf,lambda,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta,reflectance,onePlusLambda_V);
}

// anisotropic PDF
float beckmann_pdf_wo_clamps(in bool transmitted, in float reflectance, in float ndf, in float absNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float VdotH, in float LdotH, in float VdotHLdotH, in float ax2, in float ay2, in float orientedEta, out float onePlusLambda_V)
{
    float c2 = geom_smith::beckmann::C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
    float lambda = geom_smith::beckmann::Lambda(c2);
    return geom_smith::VNDF_pdf_wo_clamps(ndf,lambda,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta,reflectance,onePlusLambda_V);
}



quotient_and_pdf_scalar beckmann_cos_quotient_and_pdf_wo_clamps(in float ndf, in bool transmitted, in float NdotL2, in float absNdotV, in float NdotV2, in float VdotH, in float LdotH, in float VdotHLdotH, in float reflectance, in float orientedEta, in float a2)
{
    float onePlusLambda_V;
    const float pdf = beckmann_pdf_wo_clamps(transmitted, reflectance, ndf, absNdotV, NdotV2, VdotH, LdotH, VdotHLdotH, a2, orientedEta, onePlusLambda_V);

    return quotient_and_pdf_scalar::create( geom_smith::beckmann::G2_over_G1(onePlusLambda_V, NdotL2, a2), pdf );
}

template <class IncomingRayDirInfo>
quotient_and_pdf_scalar beckmann_cos_quotient_and_pdf(in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Isotropic<IncomingRayDirInfo> interaction, in IsotropicMicrofacetCache _cache, in float eta, in float a2)
{
    const float ndf = ndf::beckmann(a2,_cache.NdotH2);
    
    float orientedEta, dummy;
    const bool backside = math::getOrientedEtas(orientedEta, dummy, _cache.VdotH, eta);
    const float orientedEta2 = orientedEta*orientedEta;

    const float VdotHLdotH = _cache.VdotH*_cache.LdotH;
    const bool transmitted = VdotHLdotH<0.0;
    
    const float reflectance = fresnel::dielectric_common(orientedEta2,abs(_cache.VdotH));

    const float absNdotV = abs(interaction.NdotV);

    return beckmann_cos_quotient_and_pdf_wo_clamps(ndf, transmitted, _sample.NdotL2, absNdotV, interaction.NdotV_squared, _cache.VdotH, _cache.LdotH, VdotHLdotH, reflectance, orientedEta, a2);
}


quotient_and_pdf_scalar beckmann_aniso_dielectric_cos_quotient_and_pdf_wo_clamps(in float ndf, in bool transmitted, in float NdotL2, in float TdotL2, in float BdotL2, in float absNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float VdotH, in float LdotH, in float VdotHLdotH, in float reflectance, in float orientedEta, in float ax2, in float ay2)
{
    float onePlusLambda_V;
    const float pdf = beckmann_pdf_wo_clamps(transmitted,reflectance, ndf,absNdotV,TdotV2,BdotV2,NdotV2, VdotH,LdotH,VdotHLdotH, ax2,ay2,orientedEta,onePlusLambda_V);

    return quotient_and_pdf_scalar::create( geom_smith::beckmann::G2_over_G1(onePlusLambda_V, TdotL2, BdotL2, NdotL2, ax2, ay2), pdf );
}
template <class IncomingRayDirInfo>
quotient_and_pdf_scalar beckmann_aniso_dielectric_cos_quotient_and_pdf(in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, in AnisotropicMicrofacetCache _cache, in float eta, in float ax, in float ay)
{    
    const float ax2 = ax*ax;
    const float ay2 = ay*ay;
    const float TdotH2 = _cache.TdotH*_cache.TdotH;
    const float BdotH2 = _cache.BdotH*_cache.BdotH;
    const float ndf = ndf::beckmann(ax,ay,ax2,ay2, TdotH2,BdotH2,_cache.NdotH2);

    const float TdotL2 = _sample.TdotL*_sample.TdotL;
    const float BdotL2 = _sample.BdotL*_sample.BdotL;
    
    const float TdotV2 = interaction.TdotV*interaction.TdotV;
    const float BdotV2 = interaction.BdotV*interaction.BdotV;
    
    const float VdotH = _cache.VdotH;

    float orientedEta, dummy;
    const bool backside = math::getOrientedEtas(orientedEta, dummy, VdotH, eta);
    const float orientedEta2 = orientedEta*orientedEta;
    
    const float VdotHLdotH = VdotH*_cache.LdotH;
    const bool transmitted = VdotHLdotH<0.0;
    
    const float reflectance = fresnel::dielectric_common(orientedEta2,abs(VdotH));
	return beckmann_aniso_dielectric_cos_quotient_and_pdf_wo_clamps(ndf, transmitted, _sample.NdotL2,TdotL2,BdotL2, abs(interaction.NdotV),TdotV2,BdotV2, interaction.NdotV_squared, VdotH,_cache.LdotH,VdotHLdotH, reflectance,orientedEta, ax2,ay2);
}



float beckmann_smith_height_correlated_dielectric_cos_eval_wo_clamps(
    in float NdotH2, in float NdotL2, in float absNdotV, in float NdotV2,
    in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH,
    in float orientedEta, in float orientedEta2, in float a2)
{
    const float scalar_part = brdf::specular::beckmann_height_correlated_cos_eval_DG_wo_clamps(NdotH2, NdotL2, NdotV2, a2);
    
    const float reflectance = fresnel::dielectric_common(orientedEta2,abs(VdotH));

    return reflectance*ndf::microfacet_to_light_measure_transform(scalar_part,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta);
}

// before calling you must ensure that `AnisotropicMicrofacetCache` is valid (if a given V vector can "see" the L vector)
template <class IncomingRayDirInfo>
float beckmann_smith_height_correlated_dielectric_cos_eval_wo_cache_validation(in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Isotropic<IncomingRayDirInfo> interaction, in IsotropicMicrofacetCache _cache, in float eta, in float a2)
{
    float orientedEta, dummy;
    const bool backside = math::getOrientedEtas(orientedEta, dummy, _cache.VdotH, eta);
    const float orientedEta2 = orientedEta*orientedEta;
    
    const float VdotHLdotH = _cache.VdotH*_cache.LdotH;
    const bool transmitted = VdotHLdotH<0.0;

    return beckmann_smith_height_correlated_dielectric_cos_eval_wo_clamps(
        _cache.NdotH2,_sample.NdotL2,abs(interaction.NdotV),interaction.NdotV_squared,
        transmitted,_cache.VdotH,_cache.LdotH,VdotHLdotH,
        orientedEta,orientedEta2,a2);
}

float beckmann_aniso_smith_height_correlated_dielectric_cos_eval_wo_clamps(
    in float NdotH2, in float TdotH2, in float BdotH2,
    in float NdotL2, in float TdotL2, in float BdotL2,
    in float absNdotV, in float NdotV2, in float TdotV2, in float BdotV2,
    in bool transmitted, in float VdotH, in float LdotH, in float VdotHLdotH,
    in float orientedEta, in float orientedEta2,
    in float ax, in float ax2, in float ay, in float ay2)
{
    const float scalar_part = brdf::specular::beckmann_aniso_height_correlated_cos_eval_DG_wo_clamps(NdotH2,TdotH2,BdotH2, NdotL2,TdotL2,BdotL2, NdotV2,TdotV2,BdotV2, ax, ax2, ay, ay2);
    
    const float reflectance = fresnel::dielectric_common(orientedEta2,abs(VdotH));
    
    return reflectance*ndf::microfacet_to_light_measure_transform(scalar_part,absNdotV,transmitted,VdotH,LdotH,VdotHLdotH,orientedEta);
}

// before calling you must ensure that `AnisotropicMicrofacetCache` is valid (if a given V vector can "see" the L vector)
template <class IncomingRayDirInfo>
float beckmann_aniso_smith_height_correlated_cos_eval_wo_cache_validation(in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, in AnisotropicMicrofacetCache _cache, in float eta, in float ax, in float ax2, in float ay, in float ay2)
{
    const float TdotH2 = _cache.TdotH*_cache.TdotH;
    const float BdotH2 = _cache.BdotH*_cache.BdotH;

    const float TdotL2 = _sample.TdotL*_sample.TdotL;
    const float BdotL2 = _sample.BdotL*_sample.BdotL;

    const float TdotV2 = interaction.TdotV*interaction.TdotV;
    const float BdotV2 = interaction.BdotV*interaction.BdotV;

    const float VdotH = _cache.VdotH;

    float orientedEta, dummy;
    const bool backside = math::getOrientedEtas(orientedEta, dummy, VdotH, eta);
    const float orientedEta2 = orientedEta*orientedEta;
    
    const float VdotHLdotH = VdotH*_cache.LdotH;
    const bool transmitted = VdotHLdotH<0.0;

    return beckmann_aniso_smith_height_correlated_dielectric_cos_eval_wo_clamps(
        _cache.NdotH2,TdotH2,BdotH2,
        _sample.NdotL2,TdotL2,BdotL2,
        abs(interaction.NdotV),interaction.NdotV_squared,TdotV2,BdotV2,
        transmitted,VdotH,_cache.LdotH,VdotHLdotH,
        orientedEta,orientedEta2,ax,ax*ax,ay,ay*ay);
}

}
}
}
}
}

#endif
