// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/common.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace brdf
{
namespace specular
{

template <class fresnel_t, class ndf_t, class Sample, class Interaction, class MicrofacetCache>
struct CookTorrance : MicrofacetBxDFBase<float3, float, Sample, Interaction, MicrofacetCache>
{
    fresnel_t fresnel;
    ndf::ndf_traits<ndf_t> ndf;
};

template <class IncomingRayDirInfo, class fresnel_t, class ndf_t>
struct IsotropicCookTorrance : CookTorrance<fresnel_t, ndf_t, LightSample<IncomingRayDirInfo>, surface_interactions::Isotropic<IncomingRayDirInfo>, IsotropicMicrofacetCache>
{
    using base_t = CookTorrance<fresnel_t, ndf_t, LightSample<IncomingRayDirInfo>, surface_interactions::Isotropic<IncomingRayDirInfo>, IsotropicMicrofacetCache>;

    typename base_t::spectrum_t cos_eval(
        in typename base_t::sample_t s,
        in typename base_t::interaction_t interaction,
        in typename base_t::cache_t cache)
    {
        if (interaction.NdotV > FLT_MIN)
        {
            float NG = base_t::ndf.ndf.D(cache.NdotH2);
            if (base_t::ndf.ndf.a2 > FLT_MIN)
                NG *= base_t::ndf.G2(interaction.NdotV2, s.NdotL2);

            const float3 fr = base_t::fresnel(cache.VdotH);

            return fr * base_t::ndf.dHdL(NG, max(interaction.NdotV, 0.0f));
        }
        else
            return float3(0.0, 0.0, 0.0);
    }

    typename base_t::sample_t generate(
        in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction,
        inout float3 u,
        out AnisotropicMicrofacetCache cache)
    {
        const float3 localH = base_t::ndf.ndf.generateH(interaction, u, cache);

        const float3 localV = interaction.getTangentSpaceV();
        const float3 localL = math::reflect(localV, localH, cache.VdotH);

        return typename base_t::sample_t::createTangentSpace(localV, IncomingRayDirInfo::create(localL), interaction.getTangentFrame());
    }

    typename base_t::q_pdf_t cos_quotient_and_pdf(
        in typename base_t::sample_t s,
        in typename base_t::interaction_t interaction,
        in typename base_t::cache_t cache)
    {
        float3 q = 0.0f;
        if (s.NdotL > FLT_MIN && interaction.NdotV > FLT_MIN)
        {
            const float3 reflectance = base_t::fresnel(cache.VdotH);

            float G2_over_G1 = base_t::ndf.G2_over_G1(interaction.NdotV2, s.NdotL2);
            q = reflectance * G2_over_G1;
        }
        float pdf = base_t::ndf.VNDF(cache.NdotH2, interaction.NdotV2, max(interaction.NdotV, 0.0f));

        return typename base_t::q_pdf_t::create(q, pdf);
    }
};

template <class IncomingRayDirInfo, class fresnel_t, class ndf_t>
struct AnisotropicCookTorrance : CookTorrance<fresnel_t, ndf_t, LightSample<IncomingRayDirInfo>, surface_interactions::Anisotropic<IncomingRayDirInfo>, AnisotropicMicrofacetCache>
{
    using base_t = CookTorrance<fresnel_t, ndf_t, LightSample<IncomingRayDirInfo>, surface_interactions::Anisotropic<IncomingRayDirInfo>, AnisotropicMicrofacetCache>;

    typename base_t::spectrum_t cos_eval(
        in typename base_t::sample_t s,
        in typename base_t::interaction_t interaction,
        in typename base_t::cache_t cache)
    {
        if (interaction.NdotV > FLT_MIN)
        {
            const float TdotL2 = s.TdotL * s.TdotL;
            const float BdotL2 = s.BdotL * s.BdotL;

            const float TdotV2 = interaction.TdotV * interaction.TdotV;
            const float BdotV2 = interaction.BdotV * interaction.BdotV;

            float NG = base_t::ndf.ndf.D(cache.TdotH2, cache.BdotH2, cache.NdotH2);
            if (base_t::ndf.ndf.ax > FLT_MIN || base_t::ndf.ndf.ay > FLT_MIN)
                NG *= base_t::ndf.G2(TdotV2, BdotV2, interaction.NdotV2, TdotL2, BdotL2, s.NdotL2);

            const float3 fr = base_t::fresnel(cache.VdotH);

            return fr * base_t::ndf.dHdL(NG, max(interaction.NdotV, 0.0f));
        }
        else
            return float3(0.0, 0.0, 0.0);
    }

    typename base_t::sample_t generate(
        in surface_interactions::Anisotropic<IncomingRayDirInfo> interaction,
        inout float3 u,
        out AnisotropicMicrofacetCache cache)
    {
        const float3 localH = base_t::ndf.ndf.generateH(interaction, u, cache);

        const float3 localV = interaction.getTangentSpaceV();
        const float3 localL = math::reflect(localV, localH, cache.VdotH);

        return typename base_t::sample_t::createTangentSpace(localV, IncomingRayDirInfo::create(localL), interaction.getTangentFrame());
    }

    typename base_t::q_pdf_t cos_quotient_and_pdf(
        in typename base_t::sample_t s,
        in typename base_t::interaction_t interaction,
        in typename base_t::cache_t cache)
    {
        const float TdotV2 = interaction.TdotV * interaction.TdotV;
        const float BdotV2 = interaction.BdotV * interaction.BdotV;

        float3 q = 0.0f;
        if (s.NdotL > FLT_MIN && interaction.NdotV > FLT_MIN)
        {
            const float TdotL2 = s.TdotL * s.TdotL;
            const float BdotL2 = s.BdotL * s.BdotL;

            const float3 reflectance = base_t::fresnel(cache.VdotH);

            float G2_over_G1 = base_t::ndf.G2_over_G1(TdotV2, BdotV2, interaction.NdotV2, TdotL2, BdotL2, s.NdotL2);
            q = reflectance * G2_over_G1;
        }

        const float TdotH2 = cache.TdotH * cache.TdotH;
        const float BdotH2 = cache.BdotH * cache.BdotH;

        float pdf = base_t::ndf.VNDF(
            TdotH2, BdotH2, cache.NdotH2,
            TdotV2, BdotV2, interaction.NdotV2,
            max(interaction.NdotV, 0.0f)
        );

        return typename base_t::q_pdf_t::create(q, pdf);
    }
};

}
}
}
}
}

#endif
