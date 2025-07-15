// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_GGX_INCLUDED_

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
struct GGXParams;

template<class LS, class SI, class MC, typename Scalar>
NBL_PARTIAL_REQ_TOP(!surface_interactions::Anisotropic<SI> && !AnisotropicMicrofacetCache<MC>)
struct GGXParams<LS, SI, MC, Scalar NBL_PARTIAL_REQ_BOT(!surface_interactions::Anisotropic<SI> && !AnisotropicMicrofacetCache<MC>) >
{
    using this_t = GGXParams<LS, SI, MC, Scalar>;

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
struct GGXParams<LS, SI, MC, Scalar NBL_PARTIAL_REQ_BOT(surface_interactions::Anisotropic<SI> && AnisotropicMicrofacetCache<MC>) >
{
    using this_t = GGXParams<LS, SI, MC, Scalar>;

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
struct SGGXDielectricBxDF
{
    using this_t = SGGXDielectricBxDF<LS, Iso, Aniso, IsoCache, AnisoCache, Spectrum>;
    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using monochrome_type = vector<scalar_type, 1>;

    using isotropic_interaction_type = Iso;
    using anisotropic_interaction_type = Aniso;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;
    using brdf_type = reflection::SGGXBxDF<sample_type, isotropic_interaction_type, anisotropic_interaction_type, isocache_type, anisocache_type, spectral_type>;

    using params_isotropic_t = GGXParams<LS, Iso, IsoCache, scalar_type>;
    using params_anisotropic_t = GGXParams<LS, Aniso, AnisoCache, scalar_type>;

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

    spectral_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(params.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;


        const scalar_type VdotHLdotH = params.getVdotH() * params.getLdotH();
        const bool transmitted = VdotHLdotH < 0.0;

        scalar_type NG_already_in_reflective_dL_measure;
        spectral_type dummyior;
        brdf_type ggx = brdf_type::create(A.x, dummyior, dummyior);
        typename brdf_type::params_isotropic_t brdf_params = typename brdf_type::params_isotropic_t::create(params._sample, params.interaction, params.cache, params._clamp);
        NG_already_in_reflective_dL_measure = ggx.__eval_DG_wo_clamps(brdf_params);

        ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(NG_already_in_reflective_dL_measure,params.getNdotL(),transmitted,params.getVdotH(),params.getLdotH(),VdotHLdotH,orientedEta.value[0]);
        scalar_type f = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()))[0];
        return hlsl::promote<spectral_type>(f) * microfacet_transform();
    }
    spectral_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(params.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;


        const scalar_type VdotHLdotH = params.getVdotH() * params.getLdotH();
        const bool transmitted = VdotHLdotH < 0.0;

        scalar_type NG_already_in_reflective_dL_measure;
        spectral_type dummyior;
        brdf_type ggx = brdf_type::create(A.x, A.y, dummyior, dummyior);
        typename brdf_type::params_anisotropic_t brdf_params = typename brdf_type::params_anisotropic_t::create(params._sample, params.interaction, params.cache, params._clamp);
        NG_already_in_reflective_dL_measure = ggx.__eval_DG_wo_clamps(brdf_params);

        ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_REFRACT_BIT> microfacet_transform =
            ndf::microfacet_to_light_measure_transform<ndf::GGX<scalar_type>,ndf::REFLECT_REFRACT_BIT>::create(NG_already_in_reflective_dL_measure,params.getNdotL(),transmitted,params.getVdotH(),params.getLdotH(),VdotHLdotH,orientedEta.value[0]);
        scalar_type f = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()))[0];
        return hlsl::promote<spectral_type>(f) * microfacet_transform();
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
        bxdf::ReflectRefract<scalar_type> rr = bxdf::ReflectRefract<scalar_type>::create(localV, H, rcpEta.value[0]);
        localL.direction = rr(transmitted);

        return sample_type::createFromTangentSpace(localV, localL, m);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        const vector3_type localV = interaction.getTangentSpaceV();

        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(interaction.isotropic.getNdotV(), hlsl::promote<monochrome_type>(eta));
        fresnel::OrientedEtaRcps<monochrome_type> rcpEta = fresnel::OrientedEtaRcps<monochrome_type>::create(interaction.isotropic.getNdotV(), hlsl::promote<monochrome_type>(eta));

        const vector3_type upperHemisphereV = hlsl::mix(localV, -localV, interaction.isotropic.getNdotV() < scalar_type(0.0));

        spectral_type dummyior;
        brdf_type ggx = brdf_type::create(A.x, A.y, dummyior, dummyior);
        const vector3_type H = ggx.__generate(upperHemisphereV, u.xy);

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

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(params.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;

        const scalar_type VdotHLdotH = params.getVdotH() * params.getLdotH();
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()))[0];

        scalar_type ndf, devsh_v;
        const scalar_type a2 = A.x*A.x;
        ndf::GGX<scalar_type, false> ggx_ndf;
        ndf = ggx_ndf.D(a2, params.getNdotH2());

        devsh_v = ggx_ndf.devsh_part(params.getNdotV2(), a2, 1.0-a2);
        const scalar_type lambda = ggx_ndf.G1_wo_numerator(params.getNdotV(), devsh_v);

        return ggx_ndf.DG1(ndf, lambda, transmitted, params.getVdotH(), params.getLdotH(), VdotHLdotH, orientedEta.value[0], reflectance);
    }
    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(params.getVdotH(), hlsl::promote<monochrome_type>(eta));
        const monochrome_type orientedEta2 = orientedEta.value * orientedEta.value;

        const scalar_type VdotHLdotH = params.getVdotH() * params.getLdotH();
        const bool transmitted = VdotHLdotH < 0.0;

        const scalar_type reflectance = fresnel::Dielectric<monochrome_type>::__call(orientedEta2, nbl::hlsl::abs<scalar_type>(params.getVdotH()))[0];

        scalar_type ndf, devsh_v;
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;

        ndf::GGX<scalar_type, true> ggx_ndf;
        ndf = ggx_ndf.D(A.x, A.y, ax2, ay2, params.getTdotH2(), params.getBdotH2(), params.getNdotH2());

        devsh_v = ggx_ndf.devsh_part(params.getTdotV2(), params.getBdotV2(), params.getNdotV2(), ax2, ay2);
        const scalar_type lambda = ggx_ndf.G1_wo_numerator(params.getNdotV(), devsh_v);

        return ggx_ndf.DG1(ndf, lambda, transmitted, params.getVdotH(), params.getLdotH(), VdotHLdotH, orientedEta.value[0], reflectance);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;

        scalar_type _pdf = pdf(params);

        ndf::GGX<scalar_type, false> ggx_ndf;
        scalar_type quo;
        quo = ggx_ndf.G2_over_G1(ax2, params.getNdotV(), params.getNdotV2(), params.getNdotL(), params.getNdotL2());

        return quotient_pdf_type::create(hlsl::promote<spectral_type>(quo), _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        const scalar_type ax2 = A.x*A.x;
        const scalar_type ay2 = A.y*A.y;

        scalar_type _pdf = pdf(params);

        ndf::GGX<scalar_type, true> ggx_ndf;
        scalar_type quo;
        quo = ggx_ndf.G2_over_G1(ax2, ay2, params.getNdotV(), params.getTdotV2(), params.getBdotV2(), params.getNdotV2(), params.getNdotL(), params.getTdotL2(), params.getBdotL2(), params.getNdotL2());

        return quotient_pdf_type::create(hlsl::promote<spectral_type>(quo), _pdf);
    }

    vector2_type A;
    scalar_type eta;
};

}
}
}
}

#endif
