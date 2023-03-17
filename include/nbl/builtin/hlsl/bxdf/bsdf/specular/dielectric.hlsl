// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_BSDF_SPECULAR_DIELECTRIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BSDF_SPECULAR_DIELECTRIC_INCLUDED_

#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/common.hlsl>
#include <nbl/builtin/hlsl/bxdf/transmission.hlsl>
#include <nbl/builtin/hlsl/bxdf/fresnel.hlsl>

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

// usually `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
// its basically a set of weights that determine 
// assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
// `remainderMetadata` is a variable in which the generator function returns byproducts of sample generation that would otherwise have to be redundantly calculated in `remainder_and_pdf`
template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> thin_smooth_dielectric_cos_generate_wo_clamps(in float3 V, in float3 T, in float3 B, in float3 N, in float NdotV, in float absNdotV, inout float3 u, in float3 eta2, in float3 luminosityContributionHint, out float3 remainderMetadata)
{
    // we will only ever intersect from the outside
    const float3 reflectance = fresnel::thindielectric_infinite_scatter(fresnel::dielectric_common(eta2,absNdotV));

    // we are only allowed one choice for the entire ray, so make the probability a weighted sum
    const float reflectionProb = dot(reflectance, luminosityContributionHint);

    float rcpChoiceProb;
    const bool transmitted = math::partitionRandVariable(reflectionProb, u.z, rcpChoiceProb);
    remainderMetadata = (transmitted ? (float3(1.0,1.0,1.0)-reflectance):reflectance)*rcpChoiceProb;
    
    const float3 L = (transmitted ? float3(0.0,0.0,0.0):(N*2.0*NdotV))-V;
    return LightSample<IncomingRayDirInfo>::create(IncomingRayDirInfo::create(L), dot(V,L), T, B, N);
}


template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> thin_smooth_dielectric_cos_generate_wo_clamps(in float3 V, in float3 T, in float3 B, in float3 N, in float NdotV, in float absNdotV, inout float3 u, in float3 eta2, in float3 luminosityContributionHint)
{
    float3 dummy;
    return thin_smooth_dielectric_cos_generate_wo_clamps<IncomingRayDirInfo>(V,T,B,N,NdotV,absNdotV,u,eta2,luminosityContributionHint,dummy);
}

template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> thin_smooth_dielectric_cos_generate(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, inout float3 u, in float3 eta2, in float3 luminosityContributionHint)
{
    return thin_smooth_dielectric_cos_generate_wo_clamps(interaction.V.dir,interaction.T,interaction.B,interaction.N,interaction.NdotV,abs(interaction.NdotV),u,eta2,luminosityContributionHint);
}



float3 thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(out float pdf, in float3 remainderMetadata)
{
    pdf = 1.0 / 0.0; // should be reciprocal probability of the fresnel choice divided by 0.0, but would still be an INF.
    return remainderMetadata;
}

float3 thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(out float pdf, in bool transmitted, in float absNdotV, in float3 eta2, in float3 luminosityContributionHint)
{
    const float3 reflectance = fresnel::thindielectric_infinite_scatter(fresnel::dielectric_common(eta2,absNdotV));
    const float3 sampleValue = transmitted ? (float3(1.0,1.0,1.0)-reflectance):reflectance;

    const float sampleProb = dot(sampleValue,luminosityContributionHint);

    pdf = 1.0 / 0.0;
    return thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(pdf, sampleValue / sampleProb);
}

// for information why we don't check the relation between `V` and `L` or `N` and `H`, see comments for `transmission_cos_remainder_and_pdf` in `irr/builtin/glsl/bxdf/common_samples.hlsl`
template <class IncomingRayDirInfo>
float3 thin_smooth_dielectric_cos_remainder_and_pdf(out float pdf, in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Isotropic<IncomingRayDirInfo> interaction, in float3 eta2, in float3 luminosityContributionHint)
{
    const bool transmitted = isTransmissionPath(interaction.NdotV,_sample.NdotL);
    return thin_smooth_dielectric_cos_remainder_and_pdf_wo_clamps(pdf,transmitted,abs(interaction.NdotV),eta2,luminosityContributionHint);
}



template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> smooth_dielectric_cos_generate_wo_clamps(in float3 V, in float3 T, in float3 B, in float3 N, in bool backside, in float NdotV, in float absNdotV, in float NdotV2, inout float3 u, in float rcpOrientedEta, in float orientedEta2, in float rcpOrientedEta2, out bool transmitted)
{
    const float reflectance = fresnel::dielectric_common(orientedEta2,absNdotV);

    float rcpChoiceProb;
    transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

    const float3 L = math::reflect_refract(transmitted, V, N, backside, NdotV, NdotV2, rcpOrientedEta, rcpOrientedEta2);
    return LightSample<IncomingRayDirInfo>::create(IncomingRayDirInfo::create(L), dot(V,L), T, B, N);
}

template <class IncomingRayDirInfo>
LightSample<IncomingRayDirInfo> smooth_dielectric_cos_generate(in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction, inout float3 u, in float eta)
{
    float orientedEta, rcpOrientedEta;
    const bool backside = getOrientedEtas(orientedEta, rcpOrientedEta, interaction.NdotV, eta);
    
    bool dummy;
    return smooth_dielectric_cos_generate_wo_clamps(
        interaction.V.dir,
        interaction.T,interaction.B,interaction.N,
        backside,
        interaction.NdotV,
        abs(interaction.NdotV),
        interaction.NdotV*interaction.NdotV,
        u,
        rcpOrientedEta, orientedEta*orientedEta, rcpOrientedEta*rcpOrientedEta,
        dummy
    );
}


float smooth_dielectric_cos_remainder_and_pdf(out float pdf, in bool transmitted, in float rcpOrientedEta2)
{
    pdf = 1.0 / 0.0; // should be reciprocal probability of the fresnel choice divided by 0.0, but would still be an INF.
    return transmitted ? rcpOrientedEta2:1.0;
}

// for information why we don't check the relation between `V` and `L` or `N` and `H`, see comments for `transmission_cos_remainder_and_pdf` in `irr/builtin/glsl/bxdf/common_samples.hlsl`
template <class IncomingRayDirInfo>
float smooth_dielectric_cos_remainder_and_pdf(out float pdf, in LightSample<IncomingRayDirInfo> _sample, in surface_interactions::Isotropic<IncomingRayDirInfo> interaction, in float eta)
{
    const bool transmitted = math::isTransmissionPath(interaction.NdotV,_sample.NdotL);
    
    float dummy, rcpOrientedEta;
    const bool backside = math::getOrientedEtas(dummy, rcpOrientedEta, interaction.NdotV, eta);

    return smooth_dielectric_cos_remainder_and_pdf(pdf,transmitted,rcpOrientedEta);
}

}
}
}
}
}

#endif
