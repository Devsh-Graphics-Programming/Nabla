// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_COOK_TORRANCE_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"
#include "nbl/builtin/hlsl/bxdf/fresnel.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/microfacet_to_light_transform.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{

// N (NDF), F (fresnel)
template<class Config, class N, class F, bool IsBSDF NBL_STRUCT_CONSTRAINABLE>
struct SCookTorrance;

template<class Config, class N, class F>
NBL_PARTIAL_REQ_TOP(config_concepts::MicrofacetConfiguration<Config> && ndf::NDF<N> && fresnel::Fresnel<F>)
struct SCookTorrance<Config, N, F, false NBL_PARTIAL_REQ_BOT(config_concepts::MicrofacetConfiguration<Config> && ndf::NDF<N> && fresnel::Fresnel<F>) >
{
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);

    using quant_type = typename N::quant_type;

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        if (interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            using quant_query_type = typename N::quant_query_type;
            using g2g1_query_type = typename N::g2g1_query_type;

            scalar_type dummy;
            quant_query_type qq = ndf.template createQuantQuery<isocache_type>(cache, dummy);
            g2g1_query_type gq = ndf.template createG2G1Query<sample_type, isotropic_interaction_type>(_sample, interaction);

            quant_type D = ndf.template D<sample_type, isotropic_interaction_type, isocache_type>(qq, _sample, interaction, cache);
            scalar_type DG = D.projectedLightMeasure;
            if (any<vector<bool, 2> >(ndf.__base.A > hlsl::promote<vector2_type>(numeric_limits<scalar_type>::min)))
            {
                quant_type G2 = ndf.template correlated<sample_type, isotropic_interaction_type>(gq, qq, _sample, interaction);
                DG *= G2.microfacetMeasure;
            }
            return fresnel(cache.getVdotH()) * DG;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }
    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        if (interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            using quant_query_type = typename N::quant_query_type;
            using g2g1_query_type = typename N::g2g1_query_type;

            scalar_type dummy;
            quant_query_type qq = ndf.template createQuantQuery<anisocache_type>(cache, dummy);
            g2g1_query_type gq = ndf.template createG2G1Query<sample_type, anisotropic_interaction_type>(_sample, interaction);

            quant_type D = ndf.template D<sample_type, anisotropic_interaction_type, anisocache_type>(qq, _sample, interaction, cache);
            scalar_type DG = D.projectedLightMeasure;
            if (any<vector<bool, 2> >(ndf.__base.A > hlsl::promote<vector2_type>(numeric_limits<scalar_type>::min)))
            {
                quant_type G2 = ndf.template correlated<sample_type, anisotropic_interaction_type>(gq, qq, _sample, interaction);
                DG *= G2.microfacetMeasure;
            }
            return fresnel(cache.getVdotH()) * DG;
        }
        else
            return hlsl::promote<spectral_type>(0.0);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(isocache_type) cache)
    {
        anisocache_type anisocache;
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }
    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type H = ndf.generateH(localV, u);

        cache = anisocache_type::createForReflection(localV, H);
        ray_dir_info_type localL;
        bxdf::Reflect<scalar_type> r = bxdf::Reflect<scalar_type>::create(localV, H);
        localL = localL.reflect(r);

        return sample_type::createFromTangentSpace(localL, interaction.getFromTangentSpace());
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        using quant_query_type = typename N::quant_query_type;
        using dg1_query_type = typename N::dg1_query_type;

        scalar_type dummy;
        quant_query_type qq = ndf.template createQuantQuery<isocache_type>(cache, dummy);
        dg1_query_type dq = ndf.template createDG1Query<isotropic_interaction_type, isocache_type>(interaction, cache);
        quant_type DG1 = ndf.template DG1<sample_type, isotropic_interaction_type>(dq, qq, _sample, interaction);
        return DG1.projectedLightMeasure;
    }
    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        using quant_query_type = typename N::quant_query_type;
        using dg1_query_type = typename N::dg1_query_type;

        scalar_type dummy;
        quant_query_type qq = ndf.template createQuantQuery<anisocache_type>(cache, dummy);
        dg1_query_type dq = ndf.template createDG1Query<anisotropic_interaction_type, anisocache_type>(interaction, cache);
        quant_type DG1 = ndf.template DG1<sample_type, anisotropic_interaction_type>(dq, qq, _sample, interaction);
        return DG1.projectedLightMeasure;
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type _pdf = pdf(_sample, interaction, cache);

        spectral_type quo = hlsl::promote<spectral_type>(0.0);
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            using g2g1_query_type = typename N::g2g1_query_type;

            g2g1_query_type gq = ndf.template createG2G1Query<sample_type, isotropic_interaction_type>(_sample, interaction);
            scalar_type G2_over_G1 = ndf.template G2_over_G1<sample_type, isotropic_interaction_type, isocache_type>(gq, _sample, interaction, cache);
            const spectral_type reflectance = fresnel(cache.getVdotH());
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        scalar_type _pdf = pdf(_sample, interaction, cache);

        spectral_type quo = hlsl::promote<spectral_type>(0.0);
        if (_sample.getNdotL() > numeric_limits<scalar_type>::min && interaction.getNdotV() > numeric_limits<scalar_type>::min)
        {
            using g2g1_query_type = typename N::g2g1_query_type;

            g2g1_query_type gq = ndf.template createG2G1Query<sample_type, anisotropic_interaction_type>(_sample, interaction);
            scalar_type G2_over_G1 = ndf.template G2_over_G1<sample_type, anisotropic_interaction_type, anisocache_type>(gq, _sample, interaction, cache);
            const spectral_type reflectance = fresnel(cache.getVdotH());
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }

    N ndf;
    F fresnel;
};

template<class Config, class N, class F>
NBL_PARTIAL_REQ_TOP(config_concepts::MicrofacetConfiguration<Config> && ndf::NDF<N> && fresnel::Fresnel<F>)
struct SCookTorrance<Config, N, F, true NBL_PARTIAL_REQ_BOT(config_concepts::MicrofacetConfiguration<Config> && ndf::NDF<N> && fresnel::Fresnel<F>) >
{
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(monochrome_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);

    using quant_type = typename N::quant_type;

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        using quant_query_type = typename N::quant_query_type;
        using g2g1_query_type = typename N::g2g1_query_type;

        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel.orientedEta;
        quant_query_type qq = ndf.template createQuantQuery<isocache_type>(cache, orientedEta.value[0]);
        g2g1_query_type gq = ndf.template createG2G1Query<sample_type, isotropic_interaction_type>(_sample, interaction);

        quant_type D = ndf.template D<sample_type, isotropic_interaction_type, isocache_type>(qq, _sample, interaction, cache);
        scalar_type DG = D.projectedLightMeasure;
        if (any<vector<bool, 2> >(ndf.__base.A > hlsl::promote<vector2_type>(numeric_limits<scalar_type>::min)))
        {
            quant_type G2 = ndf.template correlated<sample_type, isotropic_interaction_type>(gq, qq, _sample, interaction);
            DG *= G2.microfacetMeasure;
        }
        return hlsl::promote<spectral_type>(fresnel(hlsl::abs(cache.getVdotH()))[0]) * DG;
    }
    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        using quant_query_type = typename N::quant_query_type;
        using g2g1_query_type = typename N::g2g1_query_type;

        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel.orientedEta;
        quant_query_type qq = ndf.template createQuantQuery<anisocache_type>(cache, orientedEta.value[0]);
        g2g1_query_type gq = ndf.template createG2G1Query<sample_type, anisotropic_interaction_type>(_sample, interaction);

        quant_type D = ndf.template D<sample_type, anisotropic_interaction_type, anisocache_type>(qq, _sample, interaction, cache);
        scalar_type DG = D.projectedLightMeasure;
        if (any<vector<bool, 2> >(ndf.__base.A > hlsl::promote<vector2_type>(numeric_limits<scalar_type>::min)))
        {
            quant_type G2 = ndf.template correlated<sample_type, anisotropic_interaction_type>(gq, qq, _sample, interaction);
            DG *= G2.microfacetMeasure;
        }
        return hlsl::promote<spectral_type>(fresnel(hlsl::abs(cache.getVdotH()))[0]) * DG;
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector3_type u, NBL_REF_ARG(isocache_type) cache, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint)
    {
        anisocache_type anisocache;
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, anisocache, luminosityContributionHint);
        cache = anisocache.iso_cache;
        return s;
    }
    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type u, NBL_REF_ARG(anisocache_type) cache, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel.orientedEta;
        fresnel::OrientedEtaRcps<monochrome_type> rcpEta = orientedEta.getReciprocals();

        const vector3_type upperHemisphereV = hlsl::mix(localV, -localV, interaction.getNdotV() < scalar_type(0.0));
        const vector3_type H = ndf.generateH(upperHemisphereV, u.xy);

        const scalar_type reflectance = fresnel(hlsl::abs(hlsl::dot(localV, H)))[0];
        const scalar_type reflectionProb = hlsl::dot<spectral_type>(hlsl::promote<spectral_type>(reflectance), luminosityContributionHint);

        scalar_type rcpChoiceProb;
        scalar_type z = u.z;
        bool transmitted = math::partitionRandVariable(reflectionProb, z, rcpChoiceProb);

        cache = anisocache_type::createForReflection(localV, H);

        Refract<scalar_type> r = Refract<scalar_type>::create(localV, H);
        cache.iso_cache.LdotH = hlsl::mix(cache.getVdotH(), r.getNdotT(rcpEta.value2[0]), transmitted);
        ray_dir_info_type localL;
        bxdf::ReflectRefract<scalar_type> rr;
        rr.refract = r;
        localL = localL.reflectRefract(rr, transmitted, rcpEta.value[0]);

        // fail if samples have invalid paths
        if ((!transmitted && hlsl::sign(localL.getDirection().z) != hlsl::sign(localV.z)) || (transmitted && hlsl::sign(localL.getDirection().z) == hlsl::sign(localV.z)))
        {
            localL.direction = vector3_type(0,0,0); // should check if sample direction is invalid
        }

        return sample_type::createFromTangentSpace(localL, interaction.getFromTangentSpace());
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        using quant_query_type = typename N::quant_query_type;
        using dg1_query_type = typename N::dg1_query_type;

        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel.orientedEta;
        quant_query_type qq = ndf.template createQuantQuery<isocache_type>(cache, orientedEta.value[0]);
        dg1_query_type dq = ndf.template createDG1Query<isotropic_interaction_type, isocache_type>(interaction, cache);
        quant_type DG1 = ndf.template DG1<sample_type, isotropic_interaction_type>(dq, qq, _sample, interaction);

        const scalar_type reflectance = fresnel(hlsl::abs(cache.getVdotH()))[0];
        return hlsl::mix(reflectance, scalar_type(1.0) - reflectance, cache.isTransmission()) * DG1.projectedLightMeasure;
    }
    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        using quant_query_type = typename N::quant_query_type;
        using dg1_query_type = typename N::dg1_query_type;

        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel.orientedEta;
        quant_query_type qq = ndf.template createQuantQuery<anisocache_type>(cache, orientedEta.value[0]);
        dg1_query_type dq = ndf.template createDG1Query<anisotropic_interaction_type, anisocache_type>(interaction, cache);
        quant_type DG1 = ndf.template DG1<sample_type, anisotropic_interaction_type>(dq, qq, _sample, interaction);

        const scalar_type reflectance = fresnel(hlsl::abs(cache.getVdotH()))[0];
        return hlsl::mix(reflectance, scalar_type(1.0) - reflectance, cache.isTransmission()) * DG1.projectedLightMeasure;
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        scalar_type _pdf = pdf(_sample, interaction, cache);

        using g2g1_query_type = typename N::g2g1_query_type;

        g2g1_query_type gq = ndf.template createG2G1Query<sample_type, isotropic_interaction_type>(_sample, interaction);
        scalar_type quo = ndf.template G2_over_G1<sample_type, isotropic_interaction_type, isocache_type>(gq, _sample, interaction, cache);

        return quotient_pdf_type::create(quo, _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        scalar_type _pdf = pdf(_sample, interaction, cache);

        using g2g1_query_type = typename N::g2g1_query_type;

        g2g1_query_type gq = ndf.template createG2G1Query<sample_type, anisotropic_interaction_type>(_sample, interaction);
        scalar_type quo = ndf.template G2_over_G1<sample_type, anisotropic_interaction_type, anisocache_type>(gq, _sample, interaction, cache);

        return quotient_pdf_type::create(quo, _pdf);
    }

    N ndf;
    F fresnel;
};

}
}
}

#endif
