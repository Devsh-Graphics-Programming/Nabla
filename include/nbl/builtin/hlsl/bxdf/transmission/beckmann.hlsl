// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_BECKMANN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
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
struct SBeckmannDielectricBxDF
{
    using this_t = SBeckmannDielectricBxDF<LS, Iso, Aniso, IsoCache, AnisoCache, Spectrum>;
    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;

    using isotropic_interaction_type = Iso;
    using anisotropic_interaction_type = Aniso;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;
    using brdf_type = reflection::SBeckmannBxDF<sample_type, isotropic_interaction_type, anisotropic_interaction_type, isocache_type, anisocache_type, spectral_type>;

    using params_isotropic_t = BeckmannParams<LS, Iso, IsoCache, scalar_type>;
    using params_anisotropic_t = BeckmannParams<LS, Aniso, AnisoCache, scalar_type>;


    static this_t create(scalar_type eta, scalar_type A)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(A, A);
        return retval;
    }

    static this_t create(scalar_type eta, scalar_type ax, scalar_type ay)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(ax, ay);
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        if (params.is_aniso)
            return create(params.eta, params.A.x, params.A.y);
        else
            return create(params.eta, params.A.x);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        A = params.A;
        eta = params.eta;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(params.getVdotH(), eta);
        const scalar_type orientedEta2 = orientedEta.value * orientedEta.value;

        const scalar_type VdotHLdotH = params.getVdotH() * params.getLdotH();
        const bool transmitted = VdotHLdotH < 0.0;

        spectral_type dummyior;
        brdf_type beckmann = brdf_type::create(A.x, dummyior, dummyior);
        typename brdf_type::params_isotropic_t brdf_params = typename brdf_type::params_isotropic_t::create(params._sample, params.interaction, params.cache, params._clamp);
        const scalar_type scalar_part = beckmann.__eval_DG_wo_clamps(brdf_params);

        ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(scalar_part,params.getNdotV(),transmitted,params.getVdotH(),params.getLdotH(),VdotHLdotH,orientedEta.value);
        scalar_type f = fresnel::Dielectric<scalar_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()));
        return hlsl::promote<spectral_type>(f) * microfacet_transform();
    }
    spectral_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(params.getVdotH(), eta);
        const scalar_type orientedEta2 = orientedEta.value * orientedEta.value;

        const scalar_type VdotHLdotH = params.getVdotH() * params.getLdotH();
        const bool transmitted = VdotHLdotH < 0.0;

        spectral_type dummyior;
        brdf_type beckmann = brdf_type::create(A.x, A.y, dummyior, dummyior);
        typename brdf_type::params_anisotropic_t brdf_params = typename brdf_type::params_anisotropic_t::create(params._sample, params.interaction, params.cache, params._clamp);
        const scalar_type scalar_part = beckmann.__eval_DG_wo_clamps(brdf_params);

        ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::Beckmann<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(scalar_part,params.getNdotV(),transmitted,params.getVdotH(),params.getLdotH(),VdotHLdotH,orientedEta.value);
        scalar_type f = fresnel::Dielectric<scalar_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()));
        return hlsl::promote<spectral_type>(f) * microfacet_transform();
    }

    sample_type __generate_wo_clamps(NBL_CONST_REF_ARG(vector3_type) localV, NBL_CONST_REF_ARG(vector3_type) H, NBL_CONST_REF_ARG(matrix3x3_type) m, NBL_REF_ARG(vector3_type) u, NBL_CONST_REF_ARG(fresnel::OrientedEtas<scalar_type>) orientedEta, NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<scalar_type>) rcpEta, NBL_REF_ARG(anisocache_type) cache)
    {
        const scalar_type localVdotH = nbl::hlsl::dot<vector3_type>(localV,H);
        const scalar_type reflectance = fresnel::Dielectric<scalar_type>::__call(orientedEta.value * orientedEta.value,nbl::hlsl::abs<scalar_type>(localVdotH));
        
        scalar_type rcpChoiceProb;
        bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

        cache = anisocache_type::create(localV, H);

        const scalar_type VdotH = cache.iso_cache.getVdotH();
        Refract<scalar_type> r;
        r.recomputeNdotT(VdotH < 0.0, VdotH * VdotH, rcpEta.value2);
        cache.iso_cache.LdotH = hlsl::mix(VdotH, r.NdotT, transmitted);
        ray_dir_info_type localL;
        bxdf::ReflectRefract<scalar_type> rr = bxdf::ReflectRefract<scalar_type>::create(transmitted, localV, H, VdotH, cache.iso_cache.getLdotH(), rcpEta.value);
        localL.direction = rr(transmitted);

        return sample_type::createFromTangentSpace(localV, localL, m);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(interaction.isotropic.getNdotV(), eta);
        fresnel::OrientedEtaRcps<scalar_type> rcpEta = fresnel::OrientedEtaRcps<scalar_type>::create(interaction.isotropic.getNdotV(), eta);

        const vector3_type upperHemisphereV = orientedEta.backside ? -localV : localV;

        spectral_type dummyior;
        brdf_type beckmann = brdf_type::create(A.x, A.y, dummyior, dummyior);
        const vector3_type H = beckmann.__generate(upperHemisphereV, u.xy);

        return __generate_wo_clamps(localV, H, interaction.getFromTangentSpace(), u, orientedEta, rcpEta, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        anisocache_type dummycache;
        return generate(interaction, u, dummycache);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(isocache_type) cache)
    {
        anisocache_type anisocache;
        sample_type s = generate(anisotropic_interaction_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(params.getVdotH(), eta);
        const scalar_type orientedEta2 = orientedEta.value * orientedEta.value;

        const scalar_type VdotHLdotH = params.getVdotH() * params.getLdotH();
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnel::Dielectric<scalar_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()));
        
        scalar_type ndf, lambda;
        const scalar_type a2 = A.x*A.x;
        ndf::SIsotropicParams<scalar_type> ndfparams = ndf::SIsotropicParams<scalar_type>::create(a2, params.getNdotH(), params.getNdotH2());
        ndf::Beckmann<scalar_type> beckmann_ndf;
        ndf = beckmann_ndf(ndfparams);

        smith::Beckmann<scalar_type> beckmann_smith;
        lambda = beckmann_smith.Lambda(params.getNdotV2(), a2);

        smith::bsdf::VNDF_pdf<ndf::Beckmann<scalar_type> > vndf = smith::bsdf::VNDF_pdf<ndf::Beckmann<scalar_type> >::create(ndf, params.getNdotV());
        scalar_type _pdf = vndf(lambda,transmitted,params.getVdotH(),params.getLdotH(),VdotHLdotH,orientedEta.value,reflectance);
        onePlusLambda_V = vndf.onePlusLambda_V;

        return _pdf;
    }
    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(params.getVdotH(), eta);
        const scalar_type orientedEta2 = orientedEta.value * orientedEta.value;

        const scalar_type VdotHLdotH = params.getVdotH() * params.getLdotH();
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnel::Dielectric<scalar_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()));
        
        scalar_type ndf, lambda;
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;
        ndf::SAnisotropicParams<scalar_type> ndfparams = ndf::SAnisotropicParams<scalar_type>::create(A.x, A.y, ax2, ay2, params.getTdotH2(), params.getBdotH2(), params.getNdotH2());
        ndf::Beckmann<scalar_type> beckmann_ndf;
        ndf = beckmann_ndf(ndfparams);

        smith::Beckmann<scalar_type> beckmann_smith;
        scalar_type c2 = beckmann_smith.C2(params.getTdotV2(), params.getBdotV2(), params.getNdotV2(), ax2, ay2);
        lambda = beckmann_smith.Lambda(c2);

        smith::bsdf::VNDF_pdf<ndf::Beckmann<scalar_type> > vndf = smith::bsdf::VNDF_pdf<ndf::Beckmann<scalar_type> >::create(ndf, params.getNdotV());
        scalar_type _pdf = vndf(lambda,transmitted,params.getVdotH(),params.getLdotH(),VdotHLdotH,orientedEta.value,reflectance);
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

        scalar_type quo;
        smith::SIsotropicParams<scalar_type> smithparams = smith::SIsotropicParams<scalar_type>::create(A.x*A.x, params.getNdotV2(), params.getNdotL2(), onePlusLambda_V);
        smith::Beckmann<scalar_type> beckmann_smith;
        quo = beckmann_smith.G2_over_G1(smithparams);

        return quotient_pdf_type::create((spectral_type)(quo), _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        scalar_type onePlusLambda_V;
        scalar_type _pdf = pdf(params, onePlusLambda_V);

        scalar_type quo;
        smith::SAnisotropicParams<scalar_type> smithparams = smith::SAnisotropicParams<scalar_type>::create(A.x*A.x, A.y*A.y, params.getTdotV2(), params.getBdotV2(), params.getNdotV2(), params.getTdotL2(), params.getBdotL2(), params.getNdotL2(), onePlusLambda_V);
        smith::Beckmann<scalar_type> beckmann_smith;
        quo = beckmann_smith.G2_over_G1(smithparams);

        return quotient_pdf_type::create((spectral_type)(quo), _pdf);
    }

    vector2_type A;
    scalar_type eta;
};

}
}
}
}

#endif
