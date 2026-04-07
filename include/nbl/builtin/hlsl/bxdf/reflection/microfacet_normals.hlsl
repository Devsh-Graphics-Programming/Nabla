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

namespace impl
{
template<class RayDirInfo, class Spectrum NBL_PRIMARY_REQUIRES(ray_dir_info::Basic<RayDirInfo> && concepts::FloatingPointLikeVectorial<Spectrum>)
struct SIsotropic
{
    using this_t = SIsotropic<RayDirInfo, Spectrum>;
    using ray_dir_info_type = RayDirInfo;
    using scalar_type = typename RayDirInfo::scalar_type;
    using vector3_type = typename RayDirInfo::vector3_type;
    using spectral_type = vector3_type;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(NBL_CONST_REF_ARG(RayDirInfo) normalizedV, const vector3_type normalizedN)
    {
        this_t retval;
        retval.V = normalizedV;
        retval.N = normalizedN;
        retval.NdotV = nbl::hlsl::dot<vector3_type>(retval.N, retval.V.getDirection());
        retval.NdotV2 = retval.NdotV * retval.NdotV;
        retval.luminosityContributionHint = hlsl::promote<spectral_type>(1.0);

        return retval;
    }

    RayDirInfo getV() NBL_CONST_MEMBER_FUNC { return V; }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return N; }
    scalar_type getNdotV(BxDFClampMode _clamp = BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC
    {
        return bxdf::conditionalAbsOrMax<scalar_type>(NdotV, _clamp);
    }
    scalar_type getNdotV(BxDFClampMode _clamp = BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC
    {
        return bxdf::conditionalAbsOrMax<scalar_type>(NdotV, _clamp);
    }
    scalar_type getNdotV(BxDFClampMode _clamp = BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC
    {
        return bxdf::conditionalAbsOrMax<scalar_type>(NdotV, _clamp);
    }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return NdotV2; }

    PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return PathOrigin::PO_SENSOR; }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return luminosityContributionHint; }

    RayDirInfo V;
    vector3_type N;
    vector3_type Np;
    vector3_type Nt;
    scalar_type NdotV;
    scalar_type NdotV2;

    spectral_type luminosityContributionHint;
};

template<class IsotropicInteraction NBL_PRIMARY_REQUIRES(Isotropic<IsotropicInteraction>)
struct SAnisotropic
{
    using this_t = SAnisotropic<IsotropicInteraction>;
    using isotropic_interaction_type = IsotropicInteraction;
    using ray_dir_info_type = typename isotropic_interaction_type::ray_dir_info_type;
    using scalar_type = typename ray_dir_info_type::scalar_type;
    using vector3_type = typename ray_dir_info_type::vector3_type;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;
    using spectral_type = typename isotropic_interaction_type::spectral_type;

    // WARNING: Changed since GLSL, now arguments need to be normalized!
    static this_t create(
        NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic,
        const vector3_type normalizedT,
        const vector3_type normalizedB
    )
    {
        this_t retval;
        retval.isotropic = isotropic;

        retval.T = normalizedT;
        retval.B = normalizedB;

        retval.TdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.T);
        retval.BdotV = nbl::hlsl::dot<vector3_type>(retval.isotropic.getV().getDirection(), retval.B);

        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic, const vector3_type normalizedT)
    {
        return create(isotropic, normalizedT, cross(isotropic.getN(), normalizedT));
    }
    static this_t create(NBL_CONST_REF_ARG(isotropic_interaction_type) isotropic)
    {
        vector3_type T, B;
        math::frisvad<vector3_type>(isotropic.getN(), T, B);
        return create(isotropic, nbl::hlsl::normalize<vector3_type>(T), nbl::hlsl::normalize<vector3_type>(B));
    }

    static this_t create(NBL_CONST_REF_ARG(ray_dir_info_type) normalizedV, const vector3_type normalizedN)
    {
        isotropic_interaction_type isotropic = isotropic_interaction_type::create(normalizedV, normalizedN);
        return create(isotropic);
    }

    ray_dir_info_type getV() NBL_CONST_MEMBER_FUNC { return isotropic.getV(); }
    vector3_type getN() NBL_CONST_MEMBER_FUNC { return isotropic.getN(); }
    scalar_type getNdotV(BxDFClampMode _clamp = BxDFClampMode::BCM_NONE) NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV(_clamp); }
    scalar_type getNdotV2() NBL_CONST_MEMBER_FUNC { return isotropic.getNdotV2(); }
    PathOrigin getPathOrigin() NBL_CONST_MEMBER_FUNC { return isotropic.getPathOrigin(); }
    spectral_type getLuminosityContributionHint() NBL_CONST_MEMBER_FUNC { return isotropic.getLuminosityContributionHint(); }

    vector3_type getT() NBL_CONST_MEMBER_FUNC { return T; }
    vector3_type getB() NBL_CONST_MEMBER_FUNC { return B; }
    scalar_type getTdotV() NBL_CONST_MEMBER_FUNC { return TdotV; }
    scalar_type getTdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getTdotV(); return t*t; }
    scalar_type getBdotV() NBL_CONST_MEMBER_FUNC { return BdotV; }
    scalar_type getBdotV2() NBL_CONST_MEMBER_FUNC { const scalar_type t = getBdotV(); return t*t; }

    vector3_type getTangentSpaceV() NBL_CONST_MEMBER_FUNC { return vector3_type(TdotV, BdotV, isotropic.getNdotV()); }
    matrix3x3_type getToTangentSpace() NBL_CONST_MEMBER_FUNC { return matrix3x3_type(T, B, isotropic.getN()); }
    matrix3x3_type getFromTangentSpace() NBL_CONST_MEMBER_FUNC { return nbl::hlsl::transpose<matrix3x3_type>(matrix3x3_type(T, B, isotropic.getN())); }

    isotropic_interaction_type isotropic;
    vector3_type T;
    vector3_type B;
    scalar_type TdotV;
    scalar_type BdotV;
};
}

template<class Config, class BRDF NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SMicrofacetNormals
{
    using this_t = SMicrofacetNormals<Config, IsBSDF>;
    BXDF_CONFIG_TYPE_ALIASES(Config);

    using random_type = conditional_t<IsBSDF, vector3_type, vector2_type>;
    struct Cache {};    // TODO: cache type?
    using isocache_type = Cache;
    using anisocache_type = Cache;
    using bxdf_type = BRDF;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = conditional_value<IsBSDF, BxDFClampMode, BxDFClampMode::BCM_ABS, BxDFClampMode::BCM_MAX>::value;

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

    // N -- geometric normal
    // Np -- perturbed normal
    // Nt -- tangent normal
    static scalar_type G1(const scalar_type clampedNdotL, const scalar_type NdotNp, const scalar_type clampedNpdotL, const scalar_type clampedNtdotL)
    {
        const scalar_type sinThetaNp = hlsl::sqrt(1.0 - NdotNp * NdotNp);
        return hlsl::min(scalar_type(1.0),
            clampedNdotL * hlsl::max(scalar_type(0.0), NdotNp)
            / (clampedNpdotL + clampedNtdotL * sinThetaNp)
        );
    }

    static scalar_type lambdaP(const scalar_type NdotNp, const scalar_type NpdotV, const scalar_type NtdotV)
    {
        const scalar_type sinThetaNp = hlsl::sqrt(1.0 - NdotNp * NdotNp);
        return NpdotV / (NpdotV + NtdotV * sinThetaNp);
    }

    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return evalAndWeight(_sample, anisotropic_interaction_type::create(interaction));
    }
    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getNp();
        const vector3_type local_Np = hlsl::mul(interaction.getToTangentSpace(), Np);
        const vector3_type local_Nt = hlsl::normalize(vector3_type(-local_Np.xy, 0.0));
        const vector3_type Nt = hlsl::mul(interaction.getFromTangentSpace(), local_Nt);

        const vector3_type V = interaction.getV();
        if (interaction.getNdotNp() <= scalar_type(0.0) || (hlsl::abs(local_Np.x) < numeric_limits<scalar_type>::min && hlsl::abs(local_Np.y) < numeric_limits<scalar_type>::min))
        {
            typename bxdf_type::isotropic_interaction_type iso = typename bxdf_type::isotropic_interaction_type::create(V, interaction.getN());
            typename bxdf_type::anisotropic_interaction_type interaction_N = typename bxdf_type::anisotropic_interaction_type::create(iso);
			return nested_brdf.evalAndWeight(_sample, interaction_N);
        }

        spectral_type eval = hlsl::promote<spectral_type>(0.0);
        
        const vector3_type L = _sample.getL().getDirection();
        const scalar_type NtdotL = hlsl::dot(Nt, L);
		const scalar_type lambda_p = lambdaP(interaction.getNdotNp(), interaction.getNpdotV(), interaction.getNtdotV());
        const scalar_type shadowing = G1(_sample.getNdotL(BxDFClampMode::BCM_MAX), interaction.getNdotNp(),
                                        hlsl::max(0.0, hlsl::dot(Np, L)), hlsl::max(0.0, NtdotL));

        typename bxdf_type::isotropic_interaction_type iso_Np = typename bxdf_type::isotropic_interaction_type::create(V, Np);
        typename bxdf_type::anisotropic_interaction_type interaction_Np = typename bxdf_type::anisotropic_interaction_type::create(iso_Np);

        // i -> p -> o
        {
            sample_type sample_single_p = sample_type::create(_sample.getL(), Np);

            // TODO: what to do with cache?
            value_weight_type eval_single_p = nested_brdf.evalAndWeight(sample_single_p, interaction_Np);
            eval += eval_single_p.value() * lambda_p * shadowing;
        }

        // i -> p -> t -> o
		if (NtdotL > scalar_type(0.0))
        {
            fresnel::Reflect<scalar_type> reflectL = fresnel::Reflect<scalar_type>::create(L, Nt);
            ray_dir_info_type L_reflected;
            L_reflected.setDirection(hlsl::normalize(reflectL(NtdotL)));
            sample_type sample_double_p = sample_type::create(L_reflected, Np); // Nt?

            const scalar_type notShadowedNpMirror = scalar_type(1.0) - G1(sample_double_p.getNdotL(BxDFClampMode::BCM_MAX), interaction.getNdotNp(),
                                        hlsl::max(0.0, hlsl::dot(Np, L)), hlsl::max(0.0, NtdotL));

            value_weight_type eval_double_p = nested_brdf.evalAndWeight(sample_double_p, interaction_Np);
			eval += eval_double_p.value() * (lambda_p * notShadowedNpMirror * shadowing);
		}

        // i -> t -> p -> o
        const scalar_type NtdotV = hlsl::dot(Nt, V);
        if (NtdotV > scalar_type(0.0))
        {
            fresnel::Reflect<scalar_type> reflectV = fresnel::Reflect<scalar_type>::create(V, Nt);
            ray_dir_info_type V_reflected;
            V_reflected.setDirection(hlsl::normalize(reflectV(NtdotV)));
            sample_type sample_double_t = sample_type::create(V_reflected, Np); // Nt?

            typename bxdf_type::isotropic_interaction_type iso_reflected = typename bxdf_type::isotropic_interaction_type::create(V_reflected.getDirection(), Np);
            typename bxdf_type::anisotropic_interaction_type interaction_reflected = typename bxdf_type::anisotropic_interaction_type::create(iso_reflected);

            value_weight_type eval_double_t = nested_brdf.evalAndWeight(sample_double_t, interaction_reflected);
			eval += eval_double_t.value() * (scalar_type(1.0) - lambda_p) * shadowing;
		}

        return value_weight_type::create(eval, forwardPdf(_sample, interaction));
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const random_type u, NBL_CONST_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getNp();
        const vector3_type local_Np = hlsl::mul(interaction.getToTangentSpace(), Np);
        const vector3_type local_Nt = hlsl::normalize(vector3_type(-local_Np.xy, 0.0));
        const vector3_type Nt = hlsl::mul(interaction.getFromTangentSpace(), local_Nt);

        const vector3_type V = interaction.getV();
        if (interaction.getNdotNp() <= scalar_type(0.0) || (hlsl::abs(local_Np.x) < numeric_limits<scalar_type>::min && hlsl::abs(local_Np.y) < numeric_limits<scalar_type>::min))
        {
            typename bxdf_type::isotropic_interaction_type iso = typename bxdf_type::isotropic_interaction_type::create(V, interaction.getN());
            typename bxdf_type::anisotropic_interaction_type interaction_N = typename bxdf_type::anisotropic_interaction_type::create(iso);
			return nested_brdf.generate(_sample, interaction_N);
        }

        typename bxdf_type::isotropic_interaction_type iso_Np = typename bxdf_type::isotropic_interaction_type::create(V, Np);
        typename bxdf_type::anisotropic_interaction_type interaction_Np = typename bxdf_type::anisotropic_interaction_type::create(iso_Np);

        sample_type s;
		if (u.x < lambdaP(interaction.getNdotNp(), interaction.getNpdotV(), interaction.getNtdotV())) {
			// sample on Np
            sample_type sample_p = sample_type::create(_sample.getL(), Np);
            s = nested_brdf.generate(sample_p, interaction_Np);

            if (!s.isValid())
                return s;

            const scalar_type shadowed = G1(s.getNdotL(), interaction.getNdotNp(),
                                        hlsl::max(0.0, hlsl::dot(Np, L)), hlsl::max(0.0, hlsl::dot(Nt, L)));
            
            if (u.y > shadowed)
            {
                // if sample dir shadowed, reflect on Nt
                fresnel::Reflect<scalar_type> reflect_s = fresnel::Reflect<scalar_type>::create(s.getL().getDirection(), Nt);
                ray_dir_info_type s_reflected;
                s_reflected.setDirection(hlsl::normalize(reflect_s()));
                s = sample_type::create(s_reflected, interaction.getN());   // Np?
            }
		}
        else
        {
			// do one reflection if we start at wt
            fresnel::Reflect<scalar_type> reflect_V = fresnel::Reflect<scalar_type>::create(V, Nt);
            ray_dir_info_type V_reflected;
            V_reflected.setDirection(hlsl::normalize(reflect_V()));
            sample_type reflV = sample_type::create(V_reflected, Np);   // Np?

			// sample on wp
            typename bxdf_type::isotropic_interaction_type iso_reflected = typename bxdf_type::isotropic_interaction_type::create(reflV, Np);
            typename bxdf_type::anisotropic_interaction_type interaction_Np = typename bxdf_type::anisotropic_interaction_type::create(iso_Np);
            s = nested_brdf.generate(reflV, interaction_Np);
            if (!s.isValid())
                return s;
		}
        if (s.getNdotL() < scalar_type(0.0))
            return sample_type::createInvalid();    // for reflection
        return s;
    }
    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const random_type u, NBL_REF_ARG(isocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        return generate(anisotropic_interaction_type::create(interaction), u, _cache);
    }

    scalar_type forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return forwardPdf(_sample, anisotropic_interaction_type::create(interaction));
    }
    scalar_type forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getNp();
        const vector3_type local_Np = hlsl::mul(interaction.getToTangentSpace(), Np);
        const vector3_type local_Nt = hlsl::normalize(vector3_type(-local_Np.xy, 0.0));
        const vector3_type Nt = hlsl::mul(interaction.getFromTangentSpace(), local_Nt);

        const vector3_type V = interaction.getV();
        if (interaction.getNdotNp() <= scalar_type(0.0) || (hlsl::abs(local_Np.x) < numeric_limits<scalar_type>::min && hlsl::abs(local_Np.y) < numeric_limits<scalar_type>::min))
        {
            typename bxdf_type::isotropic_interaction_type iso = typename bxdf_type::isotropic_interaction_type::create(V, interaction.getN());
            typename bxdf_type::anisotropic_interaction_type interaction_N = typename bxdf_type::anisotropic_interaction_type::create(iso);
			return nested_brdf.forwardPdf(_sample, interaction_N);
        }

        scalar_type pdf = scalar_type(0.0);
        const scalar_type lambda_p = lambdaP(interaction.getNdotNp(), interaction.getNpdotV(), interaction.getNtdotV());
        const sample_type sample_p = sample_type::create(_sample.getL(), Np);
        
		if (lambda_p > scalar_type(0.0))
        {
            typename bxdf_type::isotropic_interaction_type iso_Np = typename bxdf_type::isotropic_interaction_type::create(V, Np);
            typename bxdf_type::anisotropic_interaction_type interaction_Np = typename bxdf_type::anisotropic_interaction_type::create(iso_Np);

            // TODO: cache?
            pdf += lambda_p * nested_brdf.forwardPdf(sample_p, interaction_Np) * G1(sample_p.getNdotL(BxDFClampMode::BCM_MAX), interaction.getNdotNp(),
                                        hlsl::max(0.0, hlsl::dot(Np, L)), hlsl::max(0.0, NtdotL));

            const vector3_type L = _sample.getL().getDirection();
            const scalar_type NtdotL = hlsl::dot(Nt, L);
			if (NtdotL > numeric_limits<scalar_type>::min)
            {
                fresnel::Reflect<scalar_type> reflectL = fresnel::Reflect<scalar_type>::create(L, Nt);
                ray_dir_info_type L_reflected;
                L_reflected.setDirection(hlsl::normalize(reflectL(NtdotL)));
                sample_type sample_reflected = sample_type::create(L_reflected, Np);

                pdf += lambda_p * nested_brdf.forwardPdf(sample_reflected, interaction_Np) * 
                        (scalar_type(1.0) - G1(sample_reflected.getNdotL(BxDFClampMode::BCM_MAX), interaction.getNdotNp(), hlsl::max(0.0, hlsl::dot(Np, L)), hlsl::max(0.0, NtdotL)));
			}
		}

        const scalar_type NtdotV = hlsl::dot(Nt, V);
		if (lambda_p < scalar_type(1.0) && NtdotV > numeric_limits<scalar_type>::min)
        {
            fresnel::Reflect<scalar_type> reflectV = fresnel::Reflect<scalar_type>::create(V, Nt);
            ray_dir_info_type V_reflected;
            V_reflected.setDirection(hlsl::normalize(reflectL(NtdotV)));
            // sample_type sample_p = sample_type::create(L, Np);    // Nt?

            typename bxdf_type::isotropic_interaction_type iso_reflected = typename bxdf_type::isotropic_interaction_type::create(V_reflected.getDirection(), Np);
            typename bxdf_type::anisotropic_interaction_type interaction_reflected = typename bxdf_type::anisotropic_interaction_type::create(iso_reflected);

            pdf += (scalar_type(1.0) - lambda_p) * nested_brdf.forwardPdf(sample_p, interaction_reflected);
		}

		return pdf;
    }

    quotient_weight_type quotientAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        return quotientAndWeight(_sample, anisotropic_interaction_type::create(interaction), _cache);
    }
    quotient_weight_type quotientAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        const vector3_type Np = interaction.getNp();
        const vector3_type local_Np = hlsl::mul(interaction.getToTangentSpace(), Np);
        const vector3_type local_Nt = hlsl::normalize(vector3_type(-local_Np.xy, 0.0));
        const vector3_type Nt = hlsl::mul(interaction.getFromTangentSpace(), local_Nt);

        const vector3_type V = interaction.getV();
        if (interaction.getNdotNp() <= scalar_type(0.0) || (hlsl::abs(local_Np.x) < numeric_limits<scalar_type>::min && hlsl::abs(local_Np.y) < numeric_limits<scalar_type>::min))
        {
            typename bxdf_type::isotropic_interaction_type iso = typename bxdf_type::isotropic_interaction_type::create(V, interaction.getN());
            typename bxdf_type::anisotropic_interaction_type interaction_N = typename bxdf_type::anisotropic_interaction_type::create(iso);
			return nested_brdf.quotientAndWeight(_sample, interaction_N);
        }

        typename bxdf_type::isotropic_interaction_type iso_Np = typename bxdf_type::isotropic_interaction_type::create(V, Np);
        typename bxdf_type::anisotropic_interaction_type interaction_Np = typename bxdf_type::anisotropic_interaction_type::create(iso_Np);

        spectral_type quo = hlsl::promote<spectral_type>(1.0);

        sample_type neg_sample;
        {
            ray_dir_info_type L;
            L.setDirection(-_sample.getL().getDirection());
            neg_sample = sample_type::create(L, Np);
        }

        // TODO: cache?
        quotient_weight_type qw = nested_brdf.quotientAndWeight(neg_sample, interaction);
		quo *= qw.quotient();

        quo *= G1(_sample.getNdotL(), interaction.getNdotNp(), hlsl::max(0.0, hlsl::dot(Np, L)), hlsl::max(0.0, hlsl::dot(Nt, L)));

        return quotient_weight_type::create(quo, forwardPdf(_sample, interaction));
    }

    bxdf_type nested_brdf;
};

}
}
}
}

#endif
