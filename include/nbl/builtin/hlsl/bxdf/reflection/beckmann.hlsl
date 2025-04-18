// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted.hlsl"
#include "nbl/builtin/hlsl/bxdf/geom_smith.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class LS, class SI, class MC, typename Scalar NBL_STRUCT_CONSTRAINABLE>
struct BeckmannParams;

template<class LS, class SI, class MC, typename Scalar>
NBL_PARTIAL_REQ_TOP(!surface_interactions::Anisotropic<SI> && !AnisotropicMicrofacetCache<MC>)
struct BeckmannParams<LS, SI, MC, Scalar NBL_PARTIAL_REQ_BOT(!surface_interactions::Anisotropic<SI> && !AnisotropicMicrofacetCache<MC>) >
{
    using this_t = BeckmannParams<LS, SI, MC, Scalar>;

    static this_t create(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(SI) interaction, NBL_CONST_REF_ARG(MC) cache, BxDFClampMode _clamp)
    {
        this_t retval;
        retval._sample = _sample;
        retval.interaction = interaction;
        retval.cache = cache;
        retval._clamp = _clamp;
        return retval;
    }

    // iso
    Scalar getNdotV() NBL_CONST_MEMBER_FUNC { return hlsl::mix(math::conditionalAbsOrMax<Scalar>(_clamp == BxDFClampMode::BCM_ABS, interaction.getNdotV(), 0.0), interaction.getNdotV(), _clamp == BxDFClampMode::BCM_NONE); }
    Scalar getNdotVUnclamped() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(); }
    Scalar getNdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV2(); }
    Scalar getNdotL() NBL_CONST_MEMBER_FUNC { return hlsl::mix(math::conditionalAbsOrMax<Scalar>(_clamp == BxDFClampMode::BCM_ABS, _sample.getNdotL(), 0.0), _sample.getNdotL(), _clamp == BxDFClampMode::BCM_NONE); }
    Scalar getNdotLUnclamped() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(); }
    Scalar getNdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL2(); }
    Scalar getVdotL() NBL_CONST_MEMBER_FUNC { return _sample.getVdotL(); }
    Scalar getNdotH() NBL_CONST_MEMBER_FUNC { return cache.getNdotH(); }
    Scalar getNdotH2() NBL_CONST_MEMBER_FUNC { return cache.getNdotH2(); }
    Scalar getVdotH() NBL_CONST_MEMBER_FUNC { return cache.getVdotH(); }
    Scalar getLdotH() NBL_CONST_MEMBER_FUNC { return cache.getLdotH(); }

    LS _sample;
    SI interaction;
    MC cache;
    BxDFClampMode _clamp;
};
template<class LS, class SI, class MC, typename Scalar>
NBL_PARTIAL_REQ_TOP(surface_interactions::Anisotropic<SI> && AnisotropicMicrofacetCache<MC>)
struct BeckmannParams<LS, SI, MC, Scalar NBL_PARTIAL_REQ_BOT(surface_interactions::Anisotropic<SI> && AnisotropicMicrofacetCache<MC>) >
{
    using this_t = BeckmannParams<LS, SI, MC, Scalar>;

    static this_t create(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(SI) interaction, NBL_CONST_REF_ARG(MC) cache, BxDFClampMode _clamp)
    {
        this_t retval;
        retval._sample = _sample;
        retval.interaction = interaction;
        retval.cache = cache;
        retval._clamp = _clamp;
        return retval;
    }

    // iso
    Scalar getNdotV() NBL_CONST_MEMBER_FUNC { return hlsl::mix(math::conditionalAbsOrMax<Scalar>(_clamp == BxDFClampMode::BCM_ABS, interaction.getNdotV(), 0.0), interaction.getNdotV(), _clamp == BxDFClampMode::BCM_NONE); }
    Scalar getNdotVUnclamped() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(); }
    Scalar getNdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV2(); }
    Scalar getNdotL() NBL_CONST_MEMBER_FUNC { return hlsl::mix(math::conditionalAbsOrMax<Scalar>(_clamp == BxDFClampMode::BCM_ABS, _sample.getNdotL(), 0.0), _sample.getNdotL(), _clamp == BxDFClampMode::BCM_NONE); }
    Scalar getNdotLUnclamped() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(); }
    Scalar getNdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL2(); }
    Scalar getVdotL() NBL_CONST_MEMBER_FUNC { return _sample.getVdotL(); }
    Scalar getNdotH() NBL_CONST_MEMBER_FUNC { return cache.getNdotH(); }
    Scalar getNdotH2() NBL_CONST_MEMBER_FUNC { return cache.getNdotH2(); }
    Scalar getVdotH() NBL_CONST_MEMBER_FUNC { return cache.getVdotH(); }
    Scalar getLdotH() NBL_CONST_MEMBER_FUNC { return cache.getLdotH(); }

    // aniso
    Scalar getTdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getTdotL() * _sample.getTdotL(); }
    Scalar getBdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getBdotL() * _sample.getBdotL(); }
    Scalar getTdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getTdotV() * interaction.getTdotV(); }
    Scalar getBdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getBdotV() * interaction.getBdotV(); }
    Scalar getTdotH2() NBL_CONST_MEMBER_FUNC {return cache.getTdotH() * cache.getTdotH(); }
    Scalar getBdotH2() NBL_CONST_MEMBER_FUNC {return cache.getBdotH() * cache.getBdotH(); }

    LS _sample;
    SI interaction;
    MC cache;
    BxDFClampMode _clamp;
};

template<class LS, class Iso, class Aniso, class IsoCache, class AnisoCache, class Spectrum NBL_PRIMARY_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso> && CreatableIsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>)
struct SBeckmannBxDF
{
    using this_t = SBeckmannBxDF<LS, Iso, Aniso, IsoCache, AnisoCache, Spectrum>;
    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;

    using isotropic_interaction_type = Iso;
    using anisotropic_interaction_type = Aniso;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    using params_isotropic_t = BeckmannParams<LS, Iso, IsoCache, scalar_type>;
    using params_anisotropic_t = BeckmannParams<LS, Aniso, AnisoCache, scalar_type>;


    // iso
    static this_t create(scalar_type A, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.A = vector2_type(A,A);
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    // aniso
    static this_t create(scalar_type ax, scalar_type ay, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.A = vector2_type(ax,ay);
        retval.ior0 = ior0;
        retval.ior1 = ior1;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        if (params.is_aniso)
            return create(params.A.x, params.A.y, params.ior0, params.ior1);
        else
            return create(params.A.x, params.ior0, params.ior1);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        A = params.A;
        ior0 = params.ior0;
        ior1 = params.ior1;
    }

    scalar_type __eval_DG_wo_clamps(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        scalar_type a2 = A.x*A.x;
        ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.getNdotH(), params.getNdotH2());
        ndf::Beckmann<scalar_type> beckmann_ndf;
        scalar_type NG = beckmann_ndf(ndfparams);
        if (a2 > numeric_limits<scalar_type>::min)
        {
            smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(a2, params.getNdotV2(), params.getNdotL2(), 0);
            smith::Beckmann<scalar_type> beckmann_smith;
            NG *= beckmann_smith.correlated(smithparams);
        }
        return NG;
    }
    scalar_type __eval_DG_wo_clamps(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;
        ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.getTdotH2(), params.getBdotH2(), params.getNdotH2());
        ndf::Beckmann<scalar_type> beckmann_ndf;
        scalar_type NG = beckmann_ndf(ndfparams);
        if (any<vector<bool, 2> >(A > (vector2_type)numeric_limits<scalar_type>::min))
        {
            smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(ax2, ay2, params.getTdotV2(), params.getBdotV2(), params.getNdotV2(), params.getTdotL2(), params.getBdotL2(), params.getNdotL2(), 0);
            smith::Beckmann<scalar_type> beckmann_smith;
            NG *= beckmann_smith.correlated(smithparams);
        }
        return NG;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        if (params.getNdotVUnclamped() > numeric_limits<scalar_type>::min)
        {
            scalar_type scalar_part = __eval_DG_wo_clamps(params);
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_BIT> microfacet_transform = ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_BIT>::create(scalar_part, params.getNdotVUnclamped());
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.getVdotH());
            return f() * microfacet_transform();
        }
        else
            return (spectral_type)0.0;
    }
    spectral_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        if (params.getNdotVUnclamped() > numeric_limits<scalar_type>::min)
        {
            scalar_type scalar_part = __eval_DG_wo_clamps(params);
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_BIT> microfacet_transform = ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_BIT>::create(scalar_part, params.getNdotVUnclamped());
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.getVdotH());
            return f() * microfacet_transform();
        }
        else
            return (spectral_type)0.0;
    }

    vector3_type __generate(NBL_CONST_REF_ARG(vector3_type) localV, NBL_CONST_REF_ARG(vector2_type) u)
    {
        //stretch
        vector3_type V = nbl::hlsl::normalize<vector3_type>(vector3_type(A.x * localV.x, A.y * localV.y, localV.z));

        vector2_type slope;
        if (V.z > 0.9999)//V.z=NdotV=cosTheta in tangent space
        {
            scalar_type r = sqrt<scalar_type>(-log<scalar_type>(1.0 - u.x));
            scalar_type sinPhi = sin<scalar_type>(2.0 * numbers::pi<scalar_type> * u.y);
            scalar_type cosPhi = cos<scalar_type>(2.0 * numbers::pi<scalar_type> * u.y);
            slope = (vector2_type)r * vector2_type(cosPhi,sinPhi);
        }
        else
        {
            scalar_type cosTheta = V.z;
            scalar_type sinTheta = sqrt<scalar_type>(1.0 - cosTheta * cosTheta);
            scalar_type tanTheta = sinTheta / cosTheta;
            scalar_type cotTheta = 1.0 / tanTheta;

            scalar_type a = -1.0;
            scalar_type c = erf<scalar_type>(cosTheta);
            scalar_type sample_x = max<scalar_type>(u.x, 1.0e-6);
            scalar_type theta = acos<scalar_type>(cosTheta);
            scalar_type fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594*theta));
            scalar_type b = c - (1.0 + c) * pow<scalar_type>(1.0-sample_x, fit);

            scalar_type normalization = 1.0 / (1.0 + c + numbers::inv_sqrtpi<scalar_type> * tanTheta * exp<scalar_type>(-cosTheta*cosTheta));

            const int ITER_THRESHOLD = 10;
            const float MAX_ACCEPTABLE_ERR = 1.0e-5;
            int it = 0;
            float value=1000.0;
            while (++it < ITER_THRESHOLD && nbl::hlsl::abs<scalar_type>(value) > MAX_ACCEPTABLE_ERR)
            {
                if (!(b >= a && b <= c))
                    b = 0.5 * (a + c);

                float invErf = erfInv<scalar_type>(b);
                value = normalization * (1.0 + b + numbers::inv_sqrtpi<scalar_type> * tanTheta * exp<scalar_type>(-invErf * invErf)) - sample_x;
                float derivative = normalization * (1.0 - invErf * cosTheta);

                if (value > 0.0)
                    c = b;
                else
                    a = b;

                b -= value/derivative;
            }
            // TODO: investigate if we can replace these two erf^-1 calls with a box muller transform
            slope.x = erfInv<scalar_type>(b);
            slope.y = erfInv<scalar_type>(2.0 * max<scalar_type>(u.y, 1.0e-6) - 1.0);
        }

        scalar_type sinTheta = sqrt<scalar_type>(1.0 - V.z*V.z);
        scalar_type cosPhi = sinTheta==0.0 ? 1.0 : clamp<scalar_type>(V.x/sinTheta, -1.0, 1.0);
        scalar_type sinPhi = sinTheta==0.0 ? 0.0 : clamp<scalar_type>(V.y/sinTheta, -1.0, 1.0);
        //rotate
        scalar_type tmp = cosPhi*slope.x - sinPhi*slope.y;
        slope.y = sinPhi*slope.x + cosPhi*slope.y;
        slope.x = tmp;

        //unstretch
        slope = vector2_type(A.x,A.y)*slope;

        return nbl::hlsl::normalize<vector3_type>(vector3_type(-slope, 1.0));
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector2_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();
        const vector3_type H = __generate(localV, u);

        cache = anisocache_type::create(localV, H);
        ray_dir_info_type localL;
        bxdf::Reflect<scalar_type> r = bxdf::Reflect<scalar_type>::create(localV, H, cache.iso_cache.getVdotH());
        localL.direction = r();

        return sample_type::createFromTangentSpace(localV, localL, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector2_type) u, NBL_REF_ARG(isocache_type) cache)
    {
        anisocache_type anisocache;
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        scalar_type ndf, lambda;
        scalar_type a2 = A.x*A.x;
        ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.getNdotH(), params.getNdotH2());
        ndf::Beckmann<scalar_type> beckmann_ndf;
        ndf = beckmann_ndf(ndfparams);

        smith::Beckmann<scalar_type> beckmann_smith;
        lambda = beckmann_smith.Lambda(params.getNdotV2(), a2);

        smith::brdf::VNDF_pdf<ndf::Beckmann<scalar_type> > vndf = smith::brdf::VNDF_pdf<ndf::Beckmann<scalar_type> >::create(ndf, params.getNdotVUnclamped());
        scalar_type _pdf = vndf(lambda);
        onePlusLambda_V = vndf.onePlusLambda_V;

        return _pdf;
    }
    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        scalar_type ndf, lambda;
        ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, A.x*A.x, A.y*A.y, params.getTdotH2(), params.getBdotH2(), params.getNdotH2());
        ndf::Beckmann<scalar_type> beckmann_ndf;
        ndf = beckmann_ndf(ndfparams);

        smith::Beckmann<scalar_type> beckmann_smith;
        const scalar_type c2 = beckmann_smith.C2(params.getTdotV2(), params.getBdotV2(), params.getNdotV2(), A.x, A.y);
        lambda = beckmann_smith.Lambda(c2);

        smith::brdf::VNDF_pdf<ndf::Beckmann<scalar_type> > vndf = smith::brdf::VNDF_pdf<ndf::Beckmann<scalar_type> >::create(ndf, params.getNdotVUnclamped());
        scalar_type _pdf = vndf(lambda);
        onePlusLambda_V = vndf.onePlusLambda_V;

        return _pdf;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        scalar_type dummy;
        return pdf(params, dummy);
    }
    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        scalar_type dummy;
        return pdf(params, dummy);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        scalar_type onePlusLambda_V;
        scalar_type _pdf = pdf(params, onePlusLambda_V);

        smith::Beckmann<scalar_type> beckmann_smith;
        spectral_type quo = (spectral_type)0.0;
        if (params.getNdotLUnclamped() > numeric_limits<scalar_type>::min && params.getNdotVUnclamped() > numeric_limits<scalar_type>::min)
        {
            smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(A.x*A.x, params.getNdotV2(), params.getNdotL2(), onePlusLambda_V);
            scalar_type G2_over_G1 = beckmann_smith.G2_over_G1(smithparams);
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.getVdotH());
            const spectral_type reflectance = f();
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        scalar_type onePlusLambda_V;
        scalar_type _pdf = pdf(params, onePlusLambda_V);

        smith::Beckmann<scalar_type> beckmann_smith;
        spectral_type quo = (spectral_type)0.0;
        if (params.getNdotLUnclamped() > numeric_limits<scalar_type>::min && params.getNdotVUnclamped() > numeric_limits<scalar_type>::min)
        {
            smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(A.x*A.x, A.y*A.y, params.getTdotV2(), params.getBdotV2(), params.getNdotV2(), params.getTdotL2(), params.getBdotL2(), params.getNdotL2(), onePlusLambda_V);
            scalar_type G2_over_G1 = beckmann_smith.G2_over_G1(smithparams);
            fresnel::Conductor<spectral_type> f = fresnel::Conductor<spectral_type>::create(ior0, ior1, params.getVdotH());
            const spectral_type reflectance = f();
            quo = reflectance * G2_over_G1;
        }

        return quotient_pdf_type::create(quo, _pdf);
    }

    vector2_type A;
    spectral_type ior0, ior1;
};

}
}
}
}

#endif
