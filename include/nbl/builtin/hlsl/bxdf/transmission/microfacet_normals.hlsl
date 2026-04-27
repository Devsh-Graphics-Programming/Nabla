// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_MICROFACET_NORMALS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_MICROFACET_NORMALS_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/microfacet_normal_shadowing.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class Config, class BRDF, ndf::PerturbedNormalShadowing P, uint16_t Order NBL_STRUCT_CONSTRAINABLE>
struct SMicrofacetNormals;

template<class Config, class BRDF, ndf::PerturbedNormalShadowing P>
NBL_PARTIAL_REQ_TOP(config_concepts::BasicConfiguration<Config>)
struct SMicrofacetNormals<Config, BRDF, P, 1 NBL_PARTIAL_REQ_BOT(config_concepts::BasicConfiguration<Config>) >
{
    using this_t = SMicrofacetNormals<Config, BRDF, P, 1>;
    BXDF_CONFIG_TYPE_ALIASES(Config);

    NBL_CONSTEXPR_STATIC_INLINE bool IsBSDF = true;    // TODO: might combine with brdf version?
    using bxdf_type = BRDF;
    using random_type = conditional_t<IsBSDF, vector3_type, vector2_type>;
    struct Cache
    {
        typename bxdf_type::isocache_type iso_cache;
        typename bxdf_type::anisocache_type aniso_cache;
        bool sampleIsShadowed;
    };
    using isocache_type = Cache;
    using anisocache_type = Cache;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    using shadowing_method_type = ndf::ShadowingMethod<scalar_type, P>;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = conditional_value<IsBSDF, BxDFClampMode, BxDFClampMode::BCM_ABS, BxDFClampMode::BCM_MAX>::value;

    // perturbed normal Np stored in interaction
    // shading normal N (geometric normal in paper) stored in bxdf
    // tangent normal Nt derived as needed
    template<class NormalsTexAccessor> // TODO: concept for accessor
    anisotropic_interaction_type buildInteraction(NBL_CONST_REF_ARG(NormalsTexAccessor) normalMap, const vector2_type uv, const matrix<scalar_type,3,3> object_to_world, const ray_dir_info_type V) NBL_CONST_MEMBER_FUNC
    {
        const matrix<scalar_type,3,3> TBN = hlsl::transpose(object_to_world);
        vector3_type localN;
        normalMap.get(localN, TBN[2], TBN[0], TBN[1]);
        // normalMap.get(localN, uv);
        localN = hlsl::promote<vector3_type>(2.0) * localN - hlsl::promote<vector3_type>(1.0);

        const vector3_type N = hlsl::normalize(hlsl::mul(object_to_world, localN));
        isotropic_interaction_type interaction = isotropic_interaction_type::create(V, N);
        interaction.luminosityContributionHint = colorspace::scRGBtoXYZ[1];
        return anisotropic_interaction_type::create(interaction);
    }

    template<typename C=bool_constant<!traits<bxdf_type>::IsMicrofacet> NBL_FUNC_REQUIRES(C::value && !traits<bxdf_type>::IsMicrofacet)
    typename bxdf_type::anisocache_type __createChildCache(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        typename bxdf_type::anisocache_type cache;
        return cache;
    }
    template<typename C=bool_constant<traits<bxdf_type>::IsMicrofacet> NBL_FUNC_REQUIRES(C::value && traits<bxdf_type>::IsMicrofacet)
    typename bxdf_type::anisocache_type __createChildCache(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        const scalar_type eta = nested_bsdf.fresnel.getRefractionOrientedEta();
        using oriented_etas_t = fresnel::OrientedEtas<monochrome_type>;
        oriented_etas_t orientedEta = oriented_etas_t::create(scalar_type(1.0), hlsl::promote<monochrome_type>(eta));
        return bxdf_type::anisocache_type::template create<anisotropic_interaction_type, sample_type>(interaction, _sample, orientedEta);
    }

    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return evalAndWeight(_sample, anisotropic_interaction_type::create(interaction));
    }
    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getN();
        const scalar_type NdotNp = hlsl::dot(shadingNormal, Np);

        const ray_dir_info_type V = interaction.getV();
        if (NdotNp > scalar_type(1.0 - 1e-5))
        {
            typename bxdf_type::isotropic_interaction_type iso = bxdf_type::isotropic_interaction_type::create(V, shadingNormal);
            iso.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_N = bxdf_type::anisotropic_interaction_type::create(iso);
            const sample_type sample_N = sample_type::create(_sample.getL(), shadingNormal);
            return nested_bsdf.evalAndWeight(sample_N, interaction_N);
        }

        const vector3_type Nt = shadowing_method_type::computeNt(Np, shadingBasis);
        const scalar_type NpdotV = interaction.getNdotV();
        const scalar_type NtdotV = hlsl::dot(Nt, V.getDirection());
        spectral_type eval = hlsl::promote<spectral_type>(0.0);

        const vector3_type L = _sample.getL().getDirection();
        const scalar_type NdotL = hlsl::dot(shadingNormal, L);
        const scalar_type NpdotL = _sample.getNdotL(BxDFClampMode::BCM_MAX);
        const scalar_type NtdotL = hlsl::dot(Nt, L);
        const scalar_type lambda_p = shadowing_method_type::lambdaP(NdotNp, hlsl::abs(NpdotV), hlsl::abs(NtdotV));
        const scalar_type shadowing = shadowing_method_type::G1(hlsl::abs(NdotL), NdotNp,
                                        NpdotL, hlsl::abs(NtdotL));

        // i -> p -> o
        {
            value_weight_type eval_p = nested_bsdf.evalAndWeight(_sample, interaction);
            eval += eval_p.value() * lambda_p * shadowing;
        }

        // i -> t -> o
        if (NtdotV > scalar_type(0.0))
        {
            sample_type sample_t = sample_type::create(_sample.getL(), Nt);

            typename bxdf_type::isotropic_interaction_type iso_t = bxdf_type::isotropic_interaction_type::create(V, Nt);
            iso_t.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_t = bxdf_type::anisotropic_interaction_type::create(iso_t);

            const scalar_type shadowing_t = shadowing_method_type::G1(hlsl::abs(NdotL), hlsl::dot(shadingNormal, Nt),
                                        NpdotL, hlsl::abs(NtdotL));

            value_weight_type eval_t = nested_bsdf.evalAndWeight(sample_t, interaction_t);
            eval += eval_t.value() * (scalar_type(1.0) - lambda_p) * shadowing_t;
        }

        anisocache_type _cache;
        _cache.aniso_cache = __createChildCache(_sample, interaction);
        return value_weight_type::create(eval, forwardPdf(_sample, interaction, _cache));
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const random_type u, NBL_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getN();
        const scalar_type NdotNp = hlsl::dot(shadingNormal, Np);
        _cache.sampleIsShadowed = false;

        const ray_dir_info_type V = interaction.getV();
        if (NdotNp > scalar_type(1.0 - 1e-5))
        {
            typename bxdf_type::isotropic_interaction_type iso = bxdf_type::isotropic_interaction_type::create(V, shadingNormal);
            iso.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_N = bxdf_type::anisotropic_interaction_type::create(iso);
            return nested_bsdf.generate(interaction_N, u, _cache.aniso_cache);
        }

        const vector3_type Nt = shadowing_method_type::computeNt(Np, shadingBasis);
        const scalar_type NpdotV = interaction.getNdotV();
        const scalar_type NtdotV = hlsl::dot(Nt, V.getDirection());

        sample_type s;
        if (u.x < shadowing_method_type::lambdaP(NdotNp, hlsl::abs(NpdotV), hlsl::abs(NtdotV)))
        {
            // sample on Np
            s = nested_bsdf.generate(interaction, u, _cache.aniso_cache);
        }
        else
        {
            // sample on Nt
            typename bxdf_type::isotropic_interaction_type iso_t = bxdf_type::isotropic_interaction_type::create(V, Nt);
            iso_t.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_t = bxdf_type::anisotropic_interaction_type::create(iso_t);
            s = nested_bsdf.generate(interaction_t, u, _cache.aniso_cache);
            _cache.sampleIsShadowed = true;
        }
        if (!s.isValid())
            return sample_type::createInvalid();
        return s;
    }
    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const random_type u, NBL_REF_ARG(isocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, _cache);
        // _cache.iso_cache = _cache.aniso_cache.isotropic; // TODO: remove?
        return s;
    }

    scalar_type forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        return forwardPdf(_sample, anisotropic_interaction_type::create(interaction), _cache);
    }
    scalar_type forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getN();
        const scalar_type NdotNp = hlsl::dot(shadingNormal, Np);

        const ray_dir_info_type V = interaction.getV();
        if (NdotNp > scalar_type(1.0 - 1e-5))
        {
            typename bxdf_type::isotropic_interaction_type iso = bxdf_type::isotropic_interaction_type::create(V, shadingNormal);
            iso.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_N = bxdf_type::anisotropic_interaction_type::create(iso);
            const sample_type sample_N = sample_type::create(_sample.getL(), shadingNormal);
            return nested_bsdf.forwardPdf(sample_N, interaction_N, __createChildCache(sample_N, interaction_N));
        }

        const vector3_type Nt = shadowing_method_type::computeNt(Np, shadingBasis);
        const scalar_type NpdotV = interaction.getNdotV();
        const scalar_type NtdotV = hlsl::dot(Nt, V.getDirection());

        scalar_type pdf = scalar_type(0.0);
        const scalar_type lambda_p = shadowing_method_type::lambdaP(NdotNp, hlsl::abs(NpdotV), hlsl::abs(NtdotV));
        
        if (lambda_p > scalar_type(0.0))
        {
            const vector3_type L = _sample.getL().getDirection();
            const scalar_type NdotL = hlsl::dot(shadingNormal, L);
            const scalar_type NpdotL = _sample.getNdotL(BxDFClampMode::BCM_MAX);
            const scalar_type NtdotL = hlsl::dot(Nt, L);
            pdf += lambda_p * nested_bsdf.forwardPdf(_sample, interaction, _cache.aniso_cache)
                * shadowing_method_type::G1(hlsl::abs(NdotL), NdotNp, NpdotL, hlsl::abs(NtdotL));
        }

        if (lambda_p < scalar_type(1.0) && NtdotV > numeric_limits<scalar_type>::min)
        {
            sample_type sample_t = sample_type::create(_sample.getL(), Nt);

            typename bxdf_type::isotropic_interaction_type iso_t = bxdf_type::isotropic_interaction_type::create(V, Nt);
            iso_t.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_t = bxdf_type::anisotropic_interaction_type::create(iso_t);

            pdf += (scalar_type(1.0) - lambda_p) * nested_bsdf.forwardPdf(sample_t, interaction_t, __createChildCache(sample_t, interaction_t));
        }

        return pdf;
    }

    quotient_weight_type quotientAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        return quotientAndWeight(_sample, anisotropic_interaction_type::create(interaction), _cache);
    }
    quotient_weight_type quotientAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getN();
        const scalar_type NdotNp = hlsl::dot(shadingNormal, Np);

        const ray_dir_info_type V = interaction.getV();
        if (NdotNp > scalar_type(1.0 - 1e-5))
        {
            typename bxdf_type::isotropic_interaction_type iso = bxdf_type::isotropic_interaction_type::create(V, shadingNormal);
            iso.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_N = bxdf_type::anisotropic_interaction_type::create(iso);
            const sample_type sample_N = sample_type::create(_sample.getL(), shadingNormal);
            typename bxdf_type::anisocache_type cache_N = __createChildCache(sample_N, interaction_N);
            return nested_bsdf.quotientAndWeight(sample_N, interaction_N, cache_N);
        }

        const vector3_type Nt = shadowing_method_type::computeNt(Np, shadingBasis);
        spectral_type quo = hlsl::promote<spectral_type>(1.0);

        const scalar_type NpdotL = _sample.getNdotL(BxDFClampMode::BCM_MAX);
        quotient_weight_type qw = nested_bsdf.quotientAndWeight(_sample, interaction, _cache.aniso_cache);
        quo *= qw.quotient();

        if (_cache.sampleIsShadowed)
        {
            const vector3_type L = _sample.getL().getDirection();
            quo *= shadowing_method_type::G1(hlsl::abs(hlsl::dot(shadingNormal, L)), hlsl::abs(hlsl::dot(shadingNormal, Nt)), NpdotL, hlsl::abs(hlsl::dot(Nt, L)));
        }

        return quotient_weight_type::create(quo, forwardPdf(_sample, interaction, _cache));
    }

    bxdf_type nested_bsdf;
    vector3_type shadingNormal;
    matrix3x3_type shadingBasis;
};


}

template<typename C, typename B, ndf::PerturbedNormalShadowing P, uint16_t O>
struct traits<bxdf::transmission::SMicrofacetNormals<C,B,P,O> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMicrofacet = false;   // should be microfacet?
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
    NBL_CONSTEXPR_STATIC_INLINE bool TractablePdf = true;
};

}
}
}

#endif
