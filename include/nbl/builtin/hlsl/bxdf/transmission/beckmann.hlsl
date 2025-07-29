// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_BECKMANN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
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
    Scalar getNdotV() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(_clamp); }
    Scalar getNdotVUnclamped() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(BxDFClampMode::BCM_NONE); }
    Scalar getNdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV2(); }
    Scalar getNdotL() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(_clamp); }
    Scalar getNdotLUnclamped() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(BxDFClampMode::BCM_NONE); }
    Scalar getNdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL2(); }
    Scalar getVdotL() NBL_CONST_MEMBER_FUNC { return cache.getVdotL(); }
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
    Scalar getNdotV() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(_clamp); }
    Scalar getNdotVUnclamped() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(BxDFClampMode::BCM_NONE); }
    Scalar getNdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV2(); }
    Scalar getNdotL() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(_clamp); }
    Scalar getNdotLUnclamped() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL(BxDFClampMode::BCM_NONE); }
    Scalar getNdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getNdotL2(); }
    Scalar getVdotL() NBL_CONST_MEMBER_FUNC { return cache.getVdotL(); }
    Scalar getNdotH() NBL_CONST_MEMBER_FUNC { return cache.getNdotH(); }
    Scalar getNdotH2() NBL_CONST_MEMBER_FUNC { return cache.getNdotH2(); }
    Scalar getVdotH() NBL_CONST_MEMBER_FUNC { return cache.getVdotH(); }
    Scalar getLdotH() NBL_CONST_MEMBER_FUNC { return cache.getLdotH(); }

    // aniso
    Scalar getTdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getTdotL2(); }
    Scalar getBdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getBdotL2(); }
    Scalar getTdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getTdotV2(); }
    Scalar getBdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getBdotV2(); }
    Scalar getTdotH2() NBL_CONST_MEMBER_FUNC { return cache.getTdotH2(); }
    Scalar getBdotH2() NBL_CONST_MEMBER_FUNC { return cache.getBdotH2(); }

    LS _sample;
    SI interaction;
    MC cache;
    BxDFClampMode _clamp;
};

template<typename T>
struct SBeckmannDG1Query
{
    using scalar_type = T;

    scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return ndf; }
    scalar_type getAbsNdotV() NBL_CONST_MEMBER_FUNC { return absNdotV; }
    scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return lambda_V; }
    bool getTransmitted() NBL_CONST_MEMBER_FUNC { return transmitted; }
    scalar_type getOrientedEta() NBL_CONST_MEMBER_FUNC { return orientedEta; }
    scalar_type getOnePlusLambdaV() NBL_CONST_MEMBER_FUNC { return onePlusLambda_V; }

    scalar_type ndf;
    scalar_type absNdotV;
    scalar_type lambda_V;
    bool transmitted;
    scalar_type orientedEta;
    scalar_type onePlusLambda_V;
};

template<typename T>
struct SBeckmannG2overG1Query
{
    using scalar_type = T;

    bool getTransmitted() NBL_CONST_MEMBER_FUNC { return transmitted; }
    scalar_type getOnePlusLambdaV() NBL_CONST_MEMBER_FUNC { return onePlusLambda_V; }

    bool transmitted;
    scalar_type onePlusLambda_V;
};

template<class Config NBL_STRUCT_CONSTRAINABLE>
struct SBeckmannDielectricAnisotropicBxDF;

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannDielectricIsotropicBxDF
{
    using this_t = SBeckmannDielectricIsotropicBxDF<Config>;
    using scalar_type = typename Config::scalar_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using monochrome_type = vector<scalar_type, 1>;
    using ray_dir_info_type = typename Config::ray_dir_info_type;

    using isotropic_interaction_type = typename Config::isotropic_interaction_type;
    using anisotropic_interaction_type = typename Config::anisotropic_interaction_type;
    using sample_type = typename Config::sample_type;
    using spectral_type = typename Config::spectral_type;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = typename Config::isocache_type;
    using anisocache_type = typename Config::anisocache_type;
    using brdf_type = reflection::SBeckmannIsotropicBxDF<Config>;

    using params_isotropic_t = BeckmannParams<sample_type, isotropic_interaction_type, isocache_type, scalar_type>;
    using params_anisotropic_t = BeckmannParams<sample_type, anisotropic_interaction_type, anisocache_type, scalar_type>;


    static this_t create(scalar_type eta, scalar_type A)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = A;
        return retval;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(params.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;

        const scalar_type VdotHLdotH = params.cache.getVdotHLdotH();
        const bool transmitted = params.cache.isTransmission();

        spectral_type dummyior;
        brdf_type beckmann = brdf_type::create(A, dummyior, dummyior);
        typename brdf_type::params_isotropic_t brdf_params = typename brdf_type::params_isotropic_t::create(params._sample, params.interaction, params.cache, params._clamp);
        const scalar_type scalar_part = beckmann.__eval_DG_wo_clamps(brdf_params);

        const scalar_type microfacet_transform = ndf::microfacet_to_light_measure_transform<scalar_type,false,ndf::MTT_REFLECT_REFRACT>::__call(scalar_part,params.getNdotV(),transmitted,params.getVdotH(),params.getLdotH(),VdotHLdotH,orientedEta.value[0]);
        const scalar_type f = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, hlsl::abs<scalar_type>(params.getVdotH()))[0];
        return hlsl::promote<spectral_type>(f) * microfacet_transform;
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(isocache_type) cache)
    {
        SBeckmannDielectricAnisotropicBxDF<Config> beckmann_aniso = SBeckmannDielectricAnisotropicBxDF<Config>::create(eta, A, A);
        anisocache_type anisocache;
        sample_type s = beckmann_aniso.generate(anisotropic_interaction_type::create(interaction), u, anisocache);
        cache = anisocache.iso_cache;
        return s;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        SBeckmannDG1Query<scalar_type> dg1_query;
        dg1_query.absNdotV = params.getNdotV();
    
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(params.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;
        dg1_query.orientedEta = orientedEta.value[0];

        dg1_query.transmitted = params.cache.isTransmission();

        const scalar_type reflectance = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()))[0];
        
        const scalar_type a2 = A*A;
        ndf::Beckmann<scalar_type, false> beckmann_ndf;
        beckmann_ndf.a2 = a2;
        dg1_query.ndf = beckmann_ndf.template D<isocache_type>(params.cache);

        dg1_query.lambda_V = beckmann_ndf.LambdaC2(params.getNdotV2());

        scalar_type dg1 = beckmann_ndf.template DG1<SBeckmannDG1Query<scalar_type>, isocache_type>(dg1_query, params.cache);
        onePlusLambda_V = dg1_query.getOnePlusLambdaV();

        return hlsl::mix(reflectance, scalar_type(1.0) - reflectance, dg1_query.getTransmitted()) * dg1;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        scalar_type dummy;
        return pdf(params, dummy);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        scalar_type onePlusLambda_V;
        scalar_type _pdf = pdf(params, onePlusLambda_V);

        ndf::Beckmann<scalar_type, false> beckmann_ndf;
        beckmann_ndf.a2 = A*A;
        SBeckmannG2overG1Query<scalar_type> query;
        query.transmitted = params.getVdotH() * params.getLdotH() < 0.0;
        query.onePlusLambda_V = onePlusLambda_V;
        scalar_type quo = beckmann_ndf.template G2_over_G1<SBeckmannG2overG1Query<scalar_type>, sample_type>(query, params._sample);

        return quotient_pdf_type::create(hlsl::promote<spectral_type>(quo), _pdf);
    }

    scalar_type A;
    scalar_type eta;
};

template<class Config>
NBL_PARTIAL_REQ_TOP(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannDielectricAnisotropicBxDF<Config NBL_PARTIAL_REQ_BOT(config_concepts::MicrofacetConfiguration<Config>) >
{
    using this_t = SBeckmannDielectricAnisotropicBxDF<Config>;
    using scalar_type = typename Config::scalar_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using monochrome_type = vector<scalar_type, 1>;
    using ray_dir_info_type = typename Config::ray_dir_info_type;

    using isotropic_interaction_type = typename Config::isotropic_interaction_type;
    using anisotropic_interaction_type = typename Config::anisotropic_interaction_type;
    using sample_type = typename Config::sample_type;
    using spectral_type = typename Config::spectral_type;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = typename Config::isocache_type;
    using anisocache_type = typename Config::anisocache_type;
    using brdf_type = reflection::SBeckmannAnisotropicBxDF<Config>;

    using params_isotropic_t = BeckmannParams<sample_type, isotropic_interaction_type, isocache_type, scalar_type>;
    using params_anisotropic_t = BeckmannParams<sample_type, anisotropic_interaction_type, anisocache_type, scalar_type>;


    static this_t create(scalar_type eta, scalar_type ax, scalar_type ay)
    {
        this_t retval;
        retval.eta = eta;
        retval.A = vector2_type(ax, ay);
        return retval;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(params.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;

        const scalar_type VdotHLdotH = params.cache.getVdotHLdotH();
        const bool transmitted = params.cache.isTransmission();

        spectral_type dummyior;
        brdf_type beckmann = brdf_type::create(A.x, A.y, dummyior, dummyior);
        typename brdf_type::params_anisotropic_t brdf_params = typename brdf_type::params_anisotropic_t::create(params._sample, params.interaction, params.cache, params._clamp);
        const scalar_type scalar_part = beckmann.__eval_DG_wo_clamps(brdf_params);

        const scalar_type microfacet_transform = ndf::microfacet_to_light_measure_transform<scalar_type,false,ndf::MTT_REFLECT_REFRACT>::__call(scalar_part,params.getNdotV(),transmitted,params.getVdotH(),params.getLdotH(),VdotHLdotH,orientedEta.value[0]);
        const scalar_type f = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()))[0];
        return hlsl::promote<spectral_type>(f) * microfacet_transform;
    }

    sample_type __generate_wo_clamps(NBL_CONST_REF_ARG(vector3_type) localV, NBL_CONST_REF_ARG(vector3_type) H, NBL_CONST_REF_ARG(matrix3x3_type) m, NBL_REF_ARG(vector3_type) u, NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEta, NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<monochrome_type>) rcpEta, NBL_REF_ARG(anisocache_type) cache)
    {
        const scalar_type localVdotH = nbl::hlsl::dot<vector3_type>(localV,H);
        const scalar_type reflectance = fresnel::Dielectric<monochrome_type>::__call(orientedEta.value * orientedEta.value,nbl::hlsl::abs<scalar_type>(localVdotH))[0];
        
        scalar_type rcpChoiceProb;
        bool transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

        cache = anisocache_type::createForReflection(localV, H);

        const scalar_type VdotH = cache.iso_cache.getVdotH();
        Refract<scalar_type> r = Refract<scalar_type>::create(localV, H);
        cache.iso_cache.LdotH = hlsl::mix(VdotH, r.getNdotT(rcpEta.value2[0]), transmitted);
        ray_dir_info_type localL;
        bxdf::ReflectRefract<scalar_type> rr;
        rr.refract = r;
        localL.direction = rr(transmitted, rcpEta.value[0]);

        return sample_type::createFromTangentSpace(localV, localL, m);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(interaction.isotropic.getNdotV(), hlsl::promote<monochrome_type>(eta));
        fresnel::OrientedEtaRcps<monochrome_type> rcpEta = fresnel::OrientedEtaRcps<monochrome_type>::create(interaction.isotropic.getNdotV(), hlsl::promote<monochrome_type>(eta));

        const vector3_type upperHemisphereV = hlsl::mix(localV, -localV, interaction.isotropic.getNdotV() < scalar_type(0.0));

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

    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params, NBL_REF_ARG(scalar_type) onePlusLambda_V)
    {
        SBeckmannDG1Query<scalar_type> dg1_query;
        dg1_query.absNdotV = params.getNdotV();

        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(params.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;
        dg1_query.orientedEta = orientedEta.value[0];

        dg1_query.transmitted = params.cache.isTransmission();

        const scalar_type reflectance = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()))[0];
        
        ndf::Beckmann<scalar_type, true> beckmann_ndf;
        beckmann_ndf.ax = A.x;
        beckmann_ndf.ay = A.y;
        dg1_query.ndf = beckmann_ndf.template D<anisocache_type>(params.cache);

        dg1_query.lambda_V = beckmann_ndf.LambdaC2(params.getTdotV2(), params.getBdotV2(), params.getNdotV2());

        scalar_type dg1 = beckmann_ndf.template DG1<SBeckmannDG1Query<scalar_type>, anisocache_type>(dg1_query, params.cache);
        onePlusLambda_V = dg1_query.getOnePlusLambdaV();

        return hlsl::mix(reflectance, scalar_type(1.0) - reflectance, dg1_query.getTransmitted()) * dg1;
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        scalar_type dummy;
        return pdf(params, dummy);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        scalar_type onePlusLambda_V;
        scalar_type _pdf = pdf(params, onePlusLambda_V);

        ndf::Beckmann<scalar_type, true> beckmann_ndf;
        beckmann_ndf.ax = A.x;
        beckmann_ndf.ay = A.y;
        SBeckmannG2overG1Query<scalar_type> query;
        query.transmitted = params.getVdotH() * params.getLdotH() < 0.0;
        query.onePlusLambda_V = onePlusLambda_V;
        scalar_type quo = beckmann_ndf.template G2_over_G1<SBeckmannG2overG1Query<scalar_type>, sample_type>(query, params._sample);

        return quotient_pdf_type::create(hlsl::promote<spectral_type>(quo), _pdf);
    }

    vector2_type A;
    scalar_type eta;
};

}

template<typename C>
struct traits<bxdf::transmission::SBeckmannDielectricIsotropicBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename C>
struct traits<bxdf::transmission::SBeckmannDielectricAnisotropicBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
