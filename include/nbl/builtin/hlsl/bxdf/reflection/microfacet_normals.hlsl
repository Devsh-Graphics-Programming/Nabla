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

enum PerturbedNormalShadowing : uint16_t
{
    PNS_SCHUSSLER,
    PNS_YINING
};

template<typename T, PerturbedNormalShadowing P>
struct ShadowingMethod;

template<typename T>
struct ShadowingMethod<T, PNS_SCHUSSLER>
{
    using scalar_type = T;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

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

    static vector3_type computeNt(const vector3_type Np, const matrix3x3_type shadingBasis)
    {
        const vector3_type local_Np = hlsl::mul(shadingBasis, Np);
        const vector3_type local_Nt = hlsl::normalize(-vector3_type(local_Np.xy, 0.0));
        return hlsl::mul(hlsl::transpose(shadingBasis), local_Nt);
    }
};

template<typename T>
struct ShadowingMethod<T, PNS_YINING>
{
    using scalar_type = T;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    static scalar_type G1(const scalar_type clampedNdotL, const scalar_type NdotNp, const scalar_type clampedNpdotL, const scalar_type clampedNtdotL)
    {
        const scalar_type g = hlsl::min(scalar_type(1.0),
            clampedNpdotL / (clampedNdotL * NdotNp)
        );
        const scalar_type g2 = g * g;
        return -g2 * g + g2 + g;
    }

    // TODO: verify maths
    // since Nt is now perpendicular to Np and not N
    // total area of surface = 1 = hypotenuse of right triangle
    // area of perturbed facet = cos(Np) = NdotNp
    // area of tangent facet = cos(Nt) = sin(Np)
    // projected area of Np onto V = area * NpdotV
    static scalar_type lambdaP(const scalar_type NdotNp, const scalar_type clampedNpdotV, const scalar_type clampedNtdotV)
    {
        const scalar_type sinThetaNp = hlsl::sqrt(hlsl::max(1.0 - NdotNp * NdotNp, 0.0));
        const scalar_type ap = clampedNpdotV * NdotNp;
        const scalar_type at = clampedNtdotV * sinThetaNp;
        return ap / (ap + at);
    }

    static vector3_type computeNt(const vector3_type Np, const matrix3x3_type shadingBasis)
    {
        const vector3_type local_Np = hlsl::mul(shadingBasis, Np);
        const vector3_type local_Nt = hlsl::normalize(vector3_type(local_Np.xy * -local_Np.z, 1.0 - local_Np.z*local_Np.z));
        return hlsl::mul(hlsl::transpose(shadingBasis), local_Nt);
    }
};

// based on Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing: https://jo.dreggn.org/home/2017_normalmap.pdf
template<class Config, class BRDF, PerturbedNormalShadowing P NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SMicrofacetNormals
{
    using this_t = SMicrofacetNormals<Config, BRDF, P>;
    BXDF_CONFIG_TYPE_ALIASES(Config);

    NBL_CONSTEXPR_STATIC_INLINE bool IsBSDF = false;    // TODO: temp, should account for bsdfs at some point
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

    using shadowing_method_type = ShadowingMethod<scalar_type, P>;

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
    static typename bxdf_type::anisocache_type __createChildCache(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        typename bxdf_type::anisocache_type cache;
        return cache;
    }
    template<typename C=bool_constant<traits<bxdf_type>::IsMicrofacet> NBL_FUNC_REQUIRES(C::value && traits<bxdf_type>::IsMicrofacet)
    static typename bxdf_type::anisocache_type __createChildCache(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return bxdf_type::anisocache_type::template createForReflection<anisotropic_interaction_type, sample_type>(interaction, _sample);
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
            return nested_brdf.evalAndWeight(sample_N, interaction_N);
        }

        const vector3_type Nt = shadowing_method_type::computeNt(Np, shadingBasis);
        const scalar_type NpdotV = interaction.getNdotV();
        const scalar_type NtdotV = hlsl::dot(Nt, V.getDirection());
        spectral_type eval = hlsl::promote<spectral_type>(0.0);

        const vector3_type L = _sample.getL().getDirection();
        const scalar_type NdotL = hlsl::dot(shadingNormal, L);
        const scalar_type NpdotL = _sample.getNdotL(BxDFClampMode::BCM_MAX);
        const scalar_type NtdotL = hlsl::dot(Nt, L);
        const scalar_type lambda_p = shadowing_method_type::lambdaP(NdotNp, hlsl::max(scalar_type(0.0), NpdotV), hlsl::max(scalar_type(0.0), NtdotV));
        const scalar_type shadowing = shadowing_method_type::G1(hlsl::max(scalar_type(0.0), NdotL), NdotNp,
                                        NpdotL, hlsl::max(scalar_type(0.0), NtdotL));

        // i -> p -> o
        {
            value_weight_type eval_single_p = nested_brdf.evalAndWeight(_sample, interaction);
            eval += eval_single_p.value() * lambda_p * shadowing;
        }

        // i -> p -> t -> o
        if (NtdotL > scalar_type(0.0))
        {
            Reflect<scalar_type> reflectL = Reflect<scalar_type>::create(L, Nt);
            ray_dir_info_type L_reflected;
            L_reflected.setDirection(hlsl::normalize(reflectL(NtdotL)));
            sample_type sample_double_p = sample_type::create(L_reflected, Np);

            const scalar_type notShadowedNpMirror = scalar_type(1.0) - shadowing_method_type::G1(hlsl::max(scalar_type(0.0), hlsl::dot(L_reflected.getDirection(), shadingNormal)), NdotNp,
                sample_double_p.getNdotL(BxDFClampMode::BCM_MAX), hlsl::max(scalar_type(0.0), hlsl::dot(L_reflected.getDirection(), Nt)));

            value_weight_type eval_double_p = nested_brdf.evalAndWeight(sample_double_p, interaction);
            eval += eval_double_p.value() * (lambda_p * notShadowedNpMirror * shadowing);
        }

        // i -> t -> p -> o
        if (NtdotV > scalar_type(0.0))
        {
            Reflect<scalar_type> reflectV = Reflect<scalar_type>::create(V.getDirection(), Nt);
            ray_dir_info_type V_reflected;
            V_reflected.setDirection(hlsl::normalize(reflectV(NtdotV)));

            typename bxdf_type::isotropic_interaction_type iso_reflected = bxdf_type::isotropic_interaction_type::create(V_reflected, Np);
            iso_reflected.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_reflected = bxdf_type::anisotropic_interaction_type::create(iso_reflected);

            value_weight_type eval_double_t = nested_brdf.evalAndWeight(_sample, interaction_reflected);
            eval += eval_double_t.value() * (scalar_type(1.0) - lambda_p) * shadowing;
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
            return nested_brdf.generate(interaction_N, u, _cache.aniso_cache);
        }

        const vector3_type Nt = shadowing_method_type::computeNt(Np, shadingBasis);
        const scalar_type NpdotV = interaction.getNdotV();
        const scalar_type NtdotV = hlsl::dot(Nt, V.getDirection());

        sample_type s;
        if (u.x < shadowing_method_type::lambdaP(NdotNp, hlsl::max(scalar_type(0.0), NpdotV), hlsl::max(scalar_type(0.0), NtdotV)))
        {
            // sample on Np
            s = nested_brdf.generate(interaction, u, _cache.aniso_cache);

            if (!s.isValid())
                return sample_type::createInvalid();

            const vector3_type L = s.getL().getDirection();
            const scalar_type shadowed = shadowing_method_type::G1(hlsl::max(scalar_type(0.0), hlsl::dot(shadingNormal, L)), NdotNp,
                s.getNdotL(BxDFClampMode::BCM_MAX), hlsl::max(scalar_type(0.0), hlsl::dot(Nt, L)));

            if (u.y > shadowed)
            {
                // if sample dir shadowed, reflect on Nt
                Reflect<scalar_type> reflect_s = Reflect<scalar_type>::create(-L, Nt);
                ray_dir_info_type s_reflected;
                s_reflected.setDirection(hlsl::normalize(reflect_s()));
                s = sample_type::create(s_reflected, Np);
                _cache.sampleIsShadowed = true;
                _cache.aniso_cache = __createChildCache(s, interaction);
            }
        }
        else
        {
            // do one reflection if we start at wt
            Reflect<scalar_type> reflect_V = Reflect<scalar_type>::create(-V.getDirection(), Nt);
            const vector3_type V_negreflected = hlsl::normalize(reflect_V());

            // sample on wp
            ray_dir_info_type Vnr;
            Vnr.setDirection(V_negreflected);
            typename bxdf_type::isotropic_interaction_type iso_negreflected = bxdf_type::isotropic_interaction_type::create(Vnr, Np);
            iso_negreflected.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_negreflected = bxdf_type::anisotropic_interaction_type::create(iso_negreflected);
            s = nested_brdf.generate(interaction_negreflected, u, _cache.aniso_cache);
            if (!s.isValid())
                return s;
        }
        if (hlsl::dot(shadingNormal, s.getL().getDirection()) < scalar_type(0.0))
            return sample_type::createInvalid();    // for reflection
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
            return nested_brdf.forwardPdf(sample_N, interaction_N, __createChildCache(sample_N, interaction_N));
        }

        const vector3_type Nt = shadowing_method_type::computeNt(Np, shadingBasis);
        const scalar_type NpdotV = interaction.getNdotV();
        const scalar_type NtdotV = hlsl::dot(Nt, V.getDirection());

        scalar_type pdf = scalar_type(0.0);
        const scalar_type lambda_p = shadowing_method_type::lambdaP(NdotNp, hlsl::max(scalar_type(0.0), NpdotV), hlsl::max(scalar_type(0.0), NtdotV));
        
        if (lambda_p > scalar_type(0.0))
        {
            const vector3_type L = _sample.getL().getDirection();
            const scalar_type NdotL = hlsl::dot(shadingNormal, L);
            const scalar_type NpdotL = _sample.getNdotL(BxDFClampMode::BCM_MAX);
            const scalar_type NtdotL = hlsl::dot(Nt, L);
            pdf += lambda_p * nested_brdf.forwardPdf(_sample, interaction, __createChildCache(_sample, interaction)) * shadowing_method_type::G1(hlsl::max(scalar_type(0.0), NdotL), NdotNp,
                NpdotL, hlsl::max(scalar_type(0.0), NtdotL));

            if (NtdotL > numeric_limits<scalar_type>::min)
            {
                Reflect<scalar_type> reflectL = Reflect<scalar_type>::create(L, Nt);
                ray_dir_info_type L_reflected;
                L_reflected.setDirection(hlsl::normalize(reflectL(NtdotL)));
                sample_type sample_reflected = sample_type::create(L_reflected, Np);

                pdf += lambda_p * nested_brdf.forwardPdf(sample_reflected, interaction, __createChildCache(sample_reflected, interaction)) * 
                        (scalar_type(1.0) - shadowing_method_type::G1(hlsl::max(scalar_type(0.0), hlsl::dot(shadingNormal, L_reflected.getDirection())), NdotNp, sample_reflected.getNdotL(BxDFClampMode::BCM_MAX), hlsl::max(scalar_type(0.0), NtdotL)));
            }
        }

        if (lambda_p < scalar_type(1.0) && NtdotV > numeric_limits<scalar_type>::min)
        {
            Reflect<scalar_type> reflectV = Reflect<scalar_type>::create(V.getDirection(), Nt);
            ray_dir_info_type V_reflected;
            V_reflected.setDirection(hlsl::normalize(reflectV(NtdotV)));

            typename bxdf_type::isotropic_interaction_type iso_reflected = bxdf_type::isotropic_interaction_type::create(V_reflected, Np);
            iso_reflected.luminosityContributionHint = interaction.getLuminosityContributionHint();
            typename bxdf_type::anisotropic_interaction_type interaction_reflected = bxdf_type::anisotropic_interaction_type::create(iso_reflected);

            pdf += (scalar_type(1.0) - lambda_p) * nested_brdf.forwardPdf(_sample, interaction_reflected, __createChildCache(_sample, interaction_reflected));
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
            return nested_brdf.quotientAndWeight(sample_N, interaction_N, cache_N);
        }

        const vector3_type Nt = shadowing_method_type::computeNt(Np, shadingBasis);
        spectral_type quo = hlsl::promote<spectral_type>(1.0);

        const scalar_type NpdotL = _sample.getNdotL(BxDFClampMode::BCM_MAX);
        quotient_weight_type qw = nested_brdf.quotientAndWeight(_sample, interaction, _cache.aniso_cache);
        quo *= qw.quotient();

        if (_cache.sampleIsShadowed)
        {
            const vector3_type L = _sample.getL().getDirection();
            quo *= shadowing_method_type::G1(hlsl::max(scalar_type(0.0), hlsl::dot(shadingNormal, L)), NdotNp, NpdotL, hlsl::max(scalar_type(0.0), hlsl::dot(Nt, L)));
        }

        return quotient_weight_type::create(quo, forwardPdf(_sample, interaction, _cache));
    }

    bxdf_type nested_brdf;
    vector3_type shadingNormal;
    matrix3x3_type shadingBasis;
};

}

template<typename C, typename B, reflection::PerturbedNormalShadowing P>
struct traits<bxdf::reflection::SMicrofacetNormals<C,B,P> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMicrofacet = false;   // should be microfacet?
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
    NBL_CONSTEXPR_STATIC_INLINE bool TractablePdf = true;
};

}
}
}

#endif
