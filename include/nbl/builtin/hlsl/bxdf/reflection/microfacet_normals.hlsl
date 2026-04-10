// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_MICROFACET_NORMALS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_MICROFACET_NORMALS_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

// based on Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing: https://jo.dreggn.org/home/2017_normalmap.pdf
template<class Config, class BRDF NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SMicrofacetNormals
{
    using this_t = SMicrofacetNormals<Config, BRDF>;
    BXDF_CONFIG_TYPE_ALIASES(Config);

    NBL_CONSTEXPR_STATIC_INLINE bool IsBSDF = false;    // TODO: temp, should account for bsdfs at some point
    using random_type = conditional_t<IsBSDF, vector3_type, vector2_type>;
    using isocache_type = typename BRDF::isocache_type;
    using anisocache_type = typename BRDF::anisocache_type;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;
    using bxdf_type = BRDF;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = conditional_value<IsBSDF, BxDFClampMode, BxDFClampMode::BCM_ABS, BxDFClampMode::BCM_MAX>::value;

    // perturbed normal Np stored in interaction
    // shading normal N (geometric normal in paper) stored in bxdf
    // tangent normal Nt derived as needed
    template<class NormalsTexAccessor> // TODO: concept for accessor
    anisotropic_interaction_type buildInteraction(NBL_CONST_REF_ARG(NormalsTexAccessor) normalMap, const vector2_type uv, const matrix<scalar_type,3,3> object_to_world, const vector3_type V) NBL_CONST_MEMBER_FUNC
    {
        vector3_type localN;
        normalMap.template get<scalar_type, 2>(localN, uv, 0);
        localN = hlsl::normalize(hlsl::promote<vector3_type>(2.0) * localN - hlsl::promote<vector3_type>(1.0));

        const vector3_type N = hlsl::mul(object_to_world, localN);
        isotropic_interaction_type interaction = isotropic_interaction_type::create(V, N);
        return anisotropic_interaction_type::create(interaction);
    }

    static scalar_type G1(const scalar_type clampedNdotL, const scalar_type NdotNp, const scalar_type clampedNpdotL, const scalar_type clampedNtdotL)
    {
        const scalar_type sinThetaNp = hlsl::sqrt(hlsl::max(1.0 - NdotNp * NdotNp, 0.0));
        return hlsl::min(scalar_type(1.0),
            clampedNdotL * hlsl::max(scalar_type(0.0), NdotNp)
            / (clampedNpdotL + clampedNtdotL * sinThetaNp)
        );
    }

    static scalar_type lambdaP(const scalar_type NdotNp, const scalar_type clampedNpdotV, const scalar_type clampedNtdotV)
    {
        const scalar_type sinThetaNp = hlsl::sqrt(hlsl::max(1.0 - NdotNp * NdotNp, 0.0));
        return clampedNpdotV / (clampedNpdotV + clampedNtdotV * sinThetaNp);
    }

    // static scalar_type lambdaP_alt(const scalar_type NdotNp, const scalar_type clampedNpdotV, const scalar_type clampedNtdotV, bool t)
    // {
    //     const scalar_type ap = clampedNpdotV / NdotNp;
    //     const scalar_type at = clampedNtdotV * hlsl::sqrt(hlsl::max(1.0 - NdotNp * NdotNp, 0.0)) / NdotNp;
    //     return t ? at / (ap + at) : ap / (ap + at);
    // }

    static vector3_type __reflect(const vector3_type N, const vector3_type I)
    {
        return hlsl::normalize(I - scalar_type(2.0) * hlsl::dot(I, N) * N);
    }

    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return evalAndWeight(_sample, anisotropic_interaction_type::create(interaction));
    }
    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getN();
        const scalar_type NdotNp = hlsl::dot(shadingNormal, Np);
        const vector3_type local_Np = hlsl::mul(shadingBasis, Np);

        const ray_dir_info_type V = interaction.getV();
        if (NdotNp <= scalar_type(0.0) || (hlsl::abs(local_Np.x) < numeric_limits<scalar_type>::min && hlsl::abs(local_Np.y) < numeric_limits<scalar_type>::min))
        {
            typename bxdf_type::isotropic_interaction_type iso = typename bxdf_type::isotropic_interaction_type::create(V, shadingNormal);
            typename bxdf_type::anisotropic_interaction_type interaction_N = typename bxdf_type::anisotropic_interaction_type::create(iso);
            const sample_type sample_N = sample_type::create(_sample.getL(), shadingNormal);
            return nested_brdf.evalAndWeight(sample_N, interaction_N);
        }

        const vector3_type local_Nt = hlsl::normalize(-vector3_type(local_Np.xy, 0.0));
        const vector3_type Nt = hlsl::mul(hlsl::transpose(shadingBasis), local_Nt);
        const scalar_type NpdotV = interaction.getNdotV();
        const scalar_type NtdotV = hlsl::dot(Nt, V.getDirection());
        spectral_type eval = hlsl::promote<spectral_type>(0.0);
        
        const vector3_type L = _sample.getL().getDirection();
        const scalar_type NdotL = hlsl::dot(shadingNormal, L);
        const scalar_type NpdotL = _sample.getNdotL(BxDFClampMode::BCM_MAX);
        const scalar_type NtdotL = hlsl::dot(Nt, L);
        const scalar_type lambda_p = lambdaP(NdotNp, hlsl::max(scalar_type(0.0), NpdotV), hlsl::max(scalar_type(0.0), NtdotV));
        const scalar_type shadowing = G1(hlsl::max(scalar_type(0.0), NdotL), NdotNp,
                                        NpdotL, hlsl::max(scalar_type(0.0), NtdotL));

        // i -> p -> o
        {
            // TODO: what to do with cache?
            value_weight_type eval_single_p = nested_brdf.evalAndWeight(_sample, interaction);
            eval += eval_single_p.value() * lambda_p * shadowing;
        }

        // i -> p -> t -> o
        if (NtdotL > scalar_type(0.0))
        {
            // Reflect<scalar_type> reflectL = Reflect<scalar_type>::create(L, Nt);
            ray_dir_info_type L_reflected;
            // L_reflected.setDirection(hlsl::normalize(reflectL(NtdotL)));
            L_reflected.setDirection(__reflect(Nt, L));
            sample_type sample_double_p = sample_type::create(L_reflected, Np); // Nt?

            const scalar_type notShadowedNpMirror = scalar_type(1.0) - G1(hlsl::max(scalar_type(0.0), hlsl::dot(L_reflected.getDirection(), shadingNormal)), NdotNp,
                sample_double_p.getNdotL(BxDFClampMode::BCM_MAX), hlsl::max(scalar_type(0.0), hlsl::dot(L_reflected.getDirection(), Nt)));

            value_weight_type eval_double_p = nested_brdf.evalAndWeight(sample_double_p, interaction);
            eval += eval_double_p.value() * (lambda_p * notShadowedNpMirror * shadowing);
        }

        // i -> t -> p -> o
        if (NtdotV > scalar_type(0.0))
        {
            // Reflect<scalar_type> reflectV = Reflect<scalar_type>::create(V.getDirection(), Nt);
            ray_dir_info_type V_reflected;
            // V_reflected.setDirection(hlsl::normalize(reflectV(NtdotV)));
            V_reflected.setDirection(__reflect(Nt, V.getDirection()));

            typename bxdf_type::isotropic_interaction_type iso_reflected = typename bxdf_type::isotropic_interaction_type::create(V_reflected, Np);
            typename bxdf_type::anisotropic_interaction_type interaction_reflected = typename bxdf_type::anisotropic_interaction_type::create(iso_reflected);

            value_weight_type eval_double_t = nested_brdf.evalAndWeight(_sample, interaction_reflected);
            eval += eval_double_t.value() * (scalar_type(1.0) - lambda_p) * shadowing;
        }

        return value_weight_type::create(eval, forwardPdf(_sample, interaction));
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const random_type u, NBL_CONST_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getN();
        const scalar_type NdotNp = hlsl::dot(shadingNormal, Np);
        const vector3_type local_Np = hlsl::mul(shadingBasis, Np);

        const ray_dir_info_type V = interaction.getV();
        if (NdotNp <= scalar_type(0.0) || (hlsl::abs(local_Np.x) < numeric_limits<scalar_type>::min && hlsl::abs(local_Np.y) < numeric_limits<scalar_type>::min))
        {
            typename bxdf_type::isotropic_interaction_type iso = typename bxdf_type::isotropic_interaction_type::create(V, shadingNormal);
            typename bxdf_type::anisotropic_interaction_type interaction_N = typename bxdf_type::anisotropic_interaction_type::create(iso);
            return nested_brdf.generate(interaction_N, u, _cache);
        }

        const vector3_type local_Nt = hlsl::normalize(-vector3_type(local_Np.xy, 0.0));
        const vector3_type Nt = hlsl::mul(hlsl::transpose(shadingBasis), local_Nt);
        const scalar_type NpdotV = interaction.getNdotV();
        const scalar_type NtdotV = hlsl::dot(Nt, V.getDirection());

        sample_type s;
        if (u.x < lambdaP(NdotNp, hlsl::max(scalar_type(0.0), NpdotV), hlsl::max(scalar_type(0.0), NtdotV)))
        {
            // sample on Np
            s = nested_brdf.generate(interaction, u, _cache);

            if (!s.isValid())
                return s;

            const vector3_type L = s.getL().getDirection();
            const scalar_type shadowed = G1(hlsl::max(scalar_type(0.0), hlsl::dot(shadingNormal, L)), NdotNp,
                s.getNdotL(BxDFClampMode::BCM_MAX), hlsl::max(scalar_type(0.0), hlsl::dot(Nt, L)));

            if (u.y > shadowed)
            {
                // if sample dir shadowed, reflect on Nt
                // Reflect<scalar_type> reflect_s = Reflect<scalar_type>::create(L, Nt);
                ray_dir_info_type s_reflected;
                // s_reflected.setDirection(hlsl::normalize(reflect_s()));
                s_reflected.setDirection(hlsl::normalize(L + scalar_type(2.0) * hlsl::dot(-L, Nt) * Nt));
                s = sample_type::create(s_reflected, Np);
                // _cache = anisocache_type::createForReflection(interaction_Np.getTangentSpaceV(), s.getTangentSpaceL(), hlsl::dot(V, s_reflected.getDirection()));    // TODO: remove
            }
        }
        else
        {
            // do one reflection if we start at wt
            // Reflect<scalar_type> reflect_V = Reflect<scalar_type>::create(V.getDirection(), Nt);
            // const vector3_type V_negreflected = -hlsl::normalize(reflect_V());
            const vector3_type V_negreflected = hlsl::normalize(V.getDirection() + scalar_type(2.0) * hlsl::dot(-V.getDirection(), Nt) * Nt);

            // sample on wp
            ray_dir_info_type Vnr;
            Vnr.setDirection(V_negreflected);
            typename bxdf_type::isotropic_interaction_type iso_negreflected = typename bxdf_type::isotropic_interaction_type::create(Vnr, Np);
            typename bxdf_type::anisotropic_interaction_type interaction_negreflected = typename bxdf_type::anisotropic_interaction_type::create(iso_negreflected);
            s = nested_brdf.generate(interaction_negreflected, u, _cache);
            if (!s.isValid())
                return s;
        }
        if (s.getNdotL() < scalar_type(0.0))
            return sample_type::createInvalid();    // for reflection
        return s;
    }
    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const random_type u, NBL_REF_ARG(isocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        anisocache_type aniso_cache;
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, aniso_cache);
        // _cache = aniso_cache.iso_cache;  // TODO: remove
        return s;
    }

    scalar_type forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return forwardPdf(_sample, anisotropic_interaction_type::create(interaction));
    }
    scalar_type forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getN();
        const scalar_type NdotNp = hlsl::dot(shadingNormal, Np);
        const vector3_type local_Np = hlsl::mul(shadingBasis, Np);

        const ray_dir_info_type V = interaction.getV();
        if (NdotNp <= scalar_type(0.0) || (hlsl::abs(local_Np.x) < numeric_limits<scalar_type>::min && hlsl::abs(local_Np.y) < numeric_limits<scalar_type>::min))
        {
            typename bxdf_type::isotropic_interaction_type iso = typename bxdf_type::isotropic_interaction_type::create(V, shadingNormal);
            typename bxdf_type::anisotropic_interaction_type interaction_N = typename bxdf_type::anisotropic_interaction_type::create(iso);
            const sample_type sample_N = sample_type::create(_sample.getL(), shadingNormal);
            return nested_brdf.forwardPdf(sample_N, interaction_N);
        }

        const vector3_type local_Nt = hlsl::normalize(-vector3_type(local_Np.xy, 0.0));
        const vector3_type Nt = hlsl::mul(hlsl::transpose(shadingBasis), local_Nt);
        const scalar_type NpdotV = interaction.getNdotV();
        const scalar_type NtdotV = hlsl::dot(Nt, V.getDirection());

        scalar_type pdf = scalar_type(0.0);
        const scalar_type lambda_p = lambdaP(NdotNp, hlsl::max(scalar_type(0.0), NpdotV), hlsl::max(scalar_type(0.0), NtdotV));
        
        if (lambda_p > scalar_type(0.0))
        {
            // TODO: cache?
            const vector3_type L = _sample.getL().getDirection();
            const scalar_type NdotL = hlsl::dot(shadingNormal, L);
            const scalar_type NpdotL = _sample.getNdotL(BxDFClampMode::BCM_MAX);
            const scalar_type NtdotL = hlsl::dot(Nt, L);
            pdf += lambda_p * nested_brdf.forwardPdf(_sample, interaction) * G1(hlsl::max(scalar_type(0.0), NdotL), NdotNp,
                NpdotL, hlsl::max(scalar_type(0.0), NtdotL));

            if (NtdotL > numeric_limits<scalar_type>::min)
            {
                // Reflect<scalar_type> reflectL = Reflect<scalar_type>::create(L, Nt);
                ray_dir_info_type L_reflected;
                // L_reflected.setDirection(hlsl::normalize(reflectL(NtdotL)));
                L_reflected.setDirection(__reflect(Nt, L));
                sample_type sample_reflected = sample_type::create(L_reflected, Np);

                pdf += lambda_p * nested_brdf.forwardPdf(sample_reflected, interaction) * 
                        (scalar_type(1.0) - G1(hlsl::max(scalar_type(0.0), NdotL), NdotNp, sample_reflected.getNdotL(BxDFClampMode::BCM_MAX), hlsl::max(scalar_type(0.0), NtdotL)));
            }
        }

        if (lambda_p < scalar_type(1.0) && NtdotV > numeric_limits<scalar_type>::min)
        {
            // Reflect<scalar_type> reflectV = Reflect<scalar_type>::create(V.getDirection(), Nt);
            ray_dir_info_type V_reflected;
            // V_reflected.setDirection(hlsl::normalize(reflectV(NtdotV)));
            V_reflected.setDirection(__reflect(Nt, V.getDirection()));
            // sample_type sample_p = sample_type::create(L, Np);    // Nt?

            typename bxdf_type::isotropic_interaction_type iso_reflected = typename bxdf_type::isotropic_interaction_type::create(V_reflected, Np);
            typename bxdf_type::anisotropic_interaction_type interaction_reflected = typename bxdf_type::anisotropic_interaction_type::create(iso_reflected);

            pdf += (scalar_type(1.0) - lambda_p) * nested_brdf.forwardPdf(_sample, interaction_reflected);
        }

        return pdf;
    }

    quotient_weight_type quotientAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        anisocache_type aniso_cache;
        quotient_weight_type quo_weight = quotientAndWeight(_sample, anisotropic_interaction_type::create(interaction), aniso_cache);
        // _cache = aniso_cache.iso_cache;  // TODO: remove
        return quo_weight;
    }
    quotient_weight_type quotientAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getN();
        const scalar_type NdotNp = hlsl::dot(shadingNormal, Np);
        const vector3_type local_Np = hlsl::mul(shadingBasis, Np);

        const ray_dir_info_type V = interaction.getV();
        if (NdotNp <= scalar_type(0.0) || (hlsl::abs(local_Np.x) < numeric_limits<scalar_type>::min && hlsl::abs(local_Np.y) < numeric_limits<scalar_type>::min))
        {
            typename bxdf_type::isotropic_interaction_type iso = typename bxdf_type::isotropic_interaction_type::create(V, shadingNormal);
            typename bxdf_type::anisotropic_interaction_type interaction_N = typename bxdf_type::anisotropic_interaction_type::create(iso);
            const sample_type sample_N = sample_type::create(_sample.getL(), shadingNormal);
            anisocache_type cache_N;// = anisocache_type::template createForReflection<anisocache_type, sample_type>(interaction_N, sample_N);  // TODO: remove
            return nested_brdf.quotientAndWeight(sample_N, interaction_N, cache_N);
        }

        const vector3_type local_Nt = hlsl::normalize(-vector3_type(local_Np.xy, 0.0));
        const vector3_type Nt = hlsl::mul(hlsl::transpose(shadingBasis), local_Nt);

        spectral_type quo = hlsl::promote<spectral_type>(1.0);

        // TODO: might need same interaction from branch in generate

        // TODO: cache?
        const scalar_type NpdotL = _sample.getNdotL(BxDFClampMode::BCM_MAX);
        quotient_weight_type qw = nested_brdf.quotientAndWeight(_sample, interaction, _cache);
        quo *= qw.quotient();

        const vector3_type L = _sample.getL().getDirection();
        quo *= G1(hlsl::max(scalar_type(0.0), hlsl::dot(shadingNormal, L)), NdotNp, NpdotL, hlsl::max(scalar_type(0.0), hlsl::dot(Nt, L)));

        return quotient_weight_type::create(quo, forwardPdf(_sample, interaction));
    }

    bxdf_type nested_brdf;
    vector3_type shadingNormal;
    matrix3x3_type shadingBasis;
};

}

template<typename C, typename B>
struct traits<bxdf::reflection::SMicrofacetNormals<C,B> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMicrofacet = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
    NBL_CONSTEXPR_STATIC_INLINE bool TractablePdf = true;
};

}
}
}

#endif
