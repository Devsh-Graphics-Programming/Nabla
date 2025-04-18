// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_SMOOTH_DIELECTRIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_SMOOTH_DIELECTRIC_INCLUDED_

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

template<class LS, class SI, typename Scalar NBL_STRUCT_CONSTRAINABLE>
struct SmoothDielectricParams;

template<class LS, class SI, typename Scalar>
NBL_PARTIAL_REQ_TOP(!surface_interactions::Anisotropic<SI>)
struct SmoothDielectricParams<LS, SI, Scalar NBL_PARTIAL_REQ_BOT(!surface_interactions::Anisotropic<SI>) >
{
    using this_t = SmoothDielectricParams<LS, SI, Scalar>;

    static this_t create(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(SI) interaction, BxDFClampMode _clamp)
    {
        this_t retval;
        retval._sample = _sample;
        retval.interaction = interaction;
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

    LS _sample;
    SI interaction;
    BxDFClampMode _clamp;
};
template<class LS, class SI, typename Scalar>
NBL_PARTIAL_REQ_TOP(surface_interactions::Anisotropic<SI>)
struct SmoothDielectricParams<LS, SI, Scalar NBL_PARTIAL_REQ_BOT(surface_interactions::Anisotropic<SI>) >
{
    using this_t = SmoothDielectricParams<LS, SI, Scalar>;

    static this_t create(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(SI) interaction, BxDFClampMode _clamp)
    {
        this_t retval;
        retval._sample = _sample;
        retval.interaction = interaction;
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

    // aniso
    Scalar getTdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getTdotL() * _sample.getTdotL(); }
    Scalar getBdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getBdotL() * _sample.getBdotL(); }
    Scalar getTdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getTdotV() * interaction.getTdotV(); }
    Scalar getBdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getBdotV() * interaction.getBdotV(); }

    LS _sample;
    SI interaction;
    BxDFClampMode _clamp;
};

template<class LS, class Iso, class Aniso, class IsoCache, class AnisoCache, class Spectrum, bool thin> // NBL_FUNC_REQUIRES(Sample<LS> && IsotropicMicrofacetCache<IsoCache> && AnisotropicMicrofacetCache<AnisoCache>) // dxc won't let me put this in
struct SSmoothDielectricBxDF;

template<class LS, class Iso, class Aniso, class IsoCache, class AnisoCache, class Spectrum>
struct SSmoothDielectricBxDF<LS, Iso, Aniso, IsoCache, AnisoCache, Spectrum, false>
{
    using this_t = SSmoothDielectricBxDF<LS, Iso, Aniso, IsoCache, AnisoCache, Spectrum, false>;
    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector3_type = vector<scalar_type, 3>;

    using isotropic_interaction_type = Iso;
    using anisotropic_interaction_type = Aniso;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    using params_isotropic_t = SmoothDielectricParams<LS, Iso, scalar_type>;
    using params_anisotropic_t = SmoothDielectricParams<LS, Aniso, scalar_type>;


    static this_t create(scalar_type eta)
    {
        this_t retval;
        retval.eta = eta;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        return create(params.eta);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        eta = params.eta;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return (spectral_type)0;
    }
    spectral_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return (spectral_type)0;
    }

    sample_type __generate_wo_clamps(NBL_CONST_REF_ARG(vector3_type) V, NBL_CONST_REF_ARG(vector3_type) T, NBL_CONST_REF_ARG(vector3_type) B, NBL_CONST_REF_ARG(vector3_type) N, scalar_type NdotV, scalar_type absNdotV, NBL_REF_ARG(vector3_type) u, NBL_CONST_REF_ARG(fresnel::OrientedEtas<scalar_type>) orientedEta, NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<scalar_type>) rcpEta, NBL_REF_ARG(bool) transmitted)
    {
        const scalar_type reflectance = fresnel::Dielectric<scalar_type>::__call(orientedEta.value*orientedEta.value, absNdotV);

        scalar_type rcpChoiceProb;
        transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

        ray_dir_info_type L;
        Refract<scalar_type> r = Refract<scalar_type>::create(rcpEta, V, N, NdotV);
        bxdf::ReflectRefract<scalar_type> rr = bxdf::ReflectRefract<scalar_type>::create(transmitted, r);
        L.direction = rr(transmitted);
        return sample_type::create(L, nbl::hlsl::dot<vector3_type>(V, L.direction), T, B, N);
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector<scalar_type, 3>) u)
    {
        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(interaction.isotropic.getNdotV(), eta);
        fresnel::OrientedEtaRcps<scalar_type> rcpEta = fresnel::OrientedEtaRcps<scalar_type>::create(interaction.isotropic.getNdotV(), eta);
        scalar_type NdotV = interaction.isotropic.getNdotV();
        bool dummy;
        return __generate_wo_clamps(interaction.isotropic.getV().getDirection(), interaction.getT(), interaction.getB(), interaction.isotropic.getN(), NdotV, 
            NdotV, u, orientedEta, rcpEta, dummy);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector<scalar_type, 3>) u)
    {
        fresnel::OrientedEtas<scalar_type> orientedEta = fresnel::OrientedEtas<scalar_type>::create(interaction.isotropic.getNdotV(), eta);
        fresnel::OrientedEtaRcps<scalar_type> rcpEta = fresnel::OrientedEtaRcps<scalar_type>::create(interaction.isotropic.getNdotV(), eta);
        scalar_type NdotV = interaction.isotropic.getNdotV();
        bool dummy;
        return __generate_wo_clamps(interaction.isotropic.getV().getDirection(), interaction.getT(), interaction.getB(), interaction.isotropic.getN(), NdotV, 
            nbl::hlsl::abs<scalar_type>(NdotV), u, orientedEta, rcpEta, dummy);
    }
    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_REF_ARG(vector<scalar_type, 3>) u)
    {
        return generate(anisotropic_interaction_type::create(interaction), u);
    }

    // eval and pdf return 0 because smooth dielectric/conductor BxDFs are dirac delta distributions, model perfectly specular objects that scatter light to only one outgoing direction
    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return 0;
    }
    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return 0;
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        const bool transmitted = ComputeMicrofacetNormal<vector3_type>::isTransmissionPath(params.getNdotVUnclamped(), params.getNdotLUnclamped());

        fresnel::OrientedEtaRcps<scalar_type> rcpOrientedEtas = fresnel::OrientedEtaRcps<scalar_type>::create(params.getNdotV(), eta);

        const scalar_type _pdf = bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
        scalar_type quo = hlsl::mix<scalar_type, bool>(1.0, rcpOrientedEtas.value, transmitted);
        return quotient_pdf_type::create((spectral_type)(quo), _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        const bool transmitted = ComputeMicrofacetNormal<vector3_type>::isTransmissionPath(params.getNdotVUnclamped(), params.getNdotLUnclamped());

        fresnel::OrientedEtaRcps<scalar_type> rcpOrientedEtas = fresnel::OrientedEtaRcps<scalar_type>::create(params.getNdotV(), eta);

        const scalar_type _pdf = bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
        scalar_type quo = hlsl::mix(1.0, rcpOrientedEtas.value, transmitted);
        return quotient_pdf_type::create((spectral_type)(quo), _pdf);
    }

    scalar_type eta;
};

template<class LS, class Iso, class Aniso, class IsoCache, class AnisoCache, class Spectrum>
struct SSmoothDielectricBxDF<LS, Iso, Aniso, IsoCache, AnisoCache, Spectrum, true>
{
    using this_t = SSmoothDielectricBxDF<LS, Iso, Aniso, IsoCache, AnisoCache, Spectrum, true>;
    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector3_type = vector<scalar_type, 3>;

    using isotropic_interaction_type = Iso;
    using anisotropic_interaction_type = Aniso;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = IsoCache;
    using anisocache_type = AnisoCache;

    using params_isotropic_t = SmoothDielectricParams<LS, Iso, scalar_type>;
    using params_anisotropic_t = SmoothDielectricParams<LS, Aniso, scalar_type>;


    static this_t create(NBL_CONST_REF_ARG(spectral_type) eta2, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint)
    {
        this_t retval;
        retval.eta2 = eta2;
        retval.luminosityContributionHint = luminosityContributionHint;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        return create(params.eta2, params.luminosityContributionHint);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        eta2 = params.eta2;
        luminosityContributionHint = params.luminosityContributionHint;
    }

    spectral_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return (spectral_type)0;
    }
    spectral_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return (spectral_type)0;
    }

    // usually `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
    // its basically a set of weights that determine
    // assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
    // `remainderMetadata` is a variable which the generator function returns byproducts of sample generation that would otherwise have to be redundantly calculated `quotient_and_pdf`
    sample_type __generate_wo_clamps(NBL_CONST_REF_ARG(vector3_type) V, NBL_CONST_REF_ARG(vector3_type) T, NBL_CONST_REF_ARG(vector3_type) B, NBL_CONST_REF_ARG(vector3_type) N, scalar_type NdotV, scalar_type absNdotV, NBL_REF_ARG(vector3_type) u, NBL_CONST_REF_ARG(spectral_type) eta2, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint, NBL_REF_ARG(spectral_type) remainderMetadata)
    {
        // we will only ever intersect from the outside
        const spectral_type reflectance = fresnel::thinDielectricInfiniteScatter<spectral_type>(fresnel::Dielectric<spectral_type>::__call(eta2,absNdotV));

        // we are only allowed one choice for the entire ray, so make the probability a weighted sum
        const scalar_type reflectionProb = nbl::hlsl::dot<spectral_type>(reflectance, luminosityContributionHint);

        scalar_type rcpChoiceProb;
        const bool transmitted = math::partitionRandVariable(reflectionProb, u.z, rcpChoiceProb);
        remainderMetadata = (transmitted ? ((spectral_type)(1.0) - reflectance) : reflectance) * rcpChoiceProb;

        ray_dir_info_type L;
        L.direction = (transmitted ? (vector3_type)(0.0) : N * 2.0f * NdotV) - V;
        return sample_type::create(L, nbl::hlsl::dot<vector3_type>(V, L.direction), T, B, N);
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector<scalar_type, 3>) u)
    {
        scalar_type NdotV = interaction.isotropic.getNdotV();
        vector3_type dummy;
        return __generate_wo_clamps(interaction.isotropic.getV().getDirection(), interaction.getT(), interaction.getB(), interaction.isotropic.getN(), NdotV, NdotV, u, eta2, luminosityContributionHint, dummy);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector<scalar_type, 3>) u)
    {
        scalar_type NdotV = interaction.isotropic.getNdotV();
        vector3_type dummy;
        return __generate_wo_clamps(interaction.isotropic.getV().getDirection(), interaction.getT(), interaction.getB(), interaction.isotropic.getN(), NdotV, nbl::hlsl::abs<scalar_type>(NdotV), u, eta2, luminosityContributionHint, dummy);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return 0;
    }
    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return 0;
    }

    // isotropic only?
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        const bool transmitted = ComputeMicrofacetNormal<vector3_type>::isTransmissionPath(params.getNdotVUnclamped(), params.getNdotLUnclamped());
        const spectral_type reflectance = fresnel::thinDielectricInfiniteScatter<spectral_type>(fresnel::Dielectric<spectral_type>::__call(eta2, params.getNdotV()));
        const spectral_type sampleValue = hlsl::mix(reflectance, (spectral_type)(1.0) - reflectance, transmitted);

        const scalar_type sampleProb = nbl::hlsl::dot<spectral_type>(sampleValue,luminosityContributionHint);

        const scalar_type _pdf = bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
        return quotient_pdf_type::create((spectral_type)(sampleValue / sampleProb), _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        const bool transmitted = ComputeMicrofacetNormal<vector3_type>::isTransmissionPath(params.getNdotVUnclamped(), params.getNdotLUnclamped());
        const spectral_type reflectance = fresnel::thinDielectricInfiniteScatter<spectral_type>(fresnel::Dielectric<spectral_type>::__call(eta2, params.getNdotV()));
        const spectral_type sampleValue = hlsl::mix(reflectance, (spectral_type)(1.0) - reflectance, transmitted);

        const scalar_type sampleProb = nbl::hlsl::dot<spectral_type>(sampleValue,luminosityContributionHint);

        const scalar_type _pdf = bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
        return quotient_pdf_type::create((spectral_type)(sampleValue / sampleProb), _pdf);
    }

    spectral_type eta2;
    spectral_type luminosityContributionHint;
};

}
}
}
}

#endif
