// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_SMOOTH_DIELECTRIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_SMOOTH_DIELECTRIC_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SSmoothDielectricBxDF
{
    using this_t = SSmoothDielectricBxDF<Config>;
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

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_ABS;

    static this_t create(scalar_type eta)
    {
        this_t retval;
        retval.eta = eta;
        return retval;
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        return hlsl::promote<spectral_type>(0);
    }
    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return hlsl::promote<spectral_type>(0);
    }

    sample_type __generate_wo_clamps(const vector3_type V, const vector3_type T, const vector3_type B, const vector3_type N, scalar_type NdotV, scalar_type absNdotV, NBL_REF_ARG(vector3_type) u, NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEta, NBL_CONST_REF_ARG(fresnel::OrientedEtaRcps<monochrome_type>) rcpEta, NBL_REF_ARG(bool) transmitted)
    {
        const scalar_type reflectance = fresnel::Dielectric<monochrome_type>::__call(orientedEta.value*orientedEta.value, absNdotV)[0];

        scalar_type rcpChoiceProb;
        transmitted = math::partitionRandVariable(reflectance, u.z, rcpChoiceProb);

        ray_dir_info_type L;
        Refract<scalar_type> r = Refract<scalar_type>::create(V, N);
        bxdf::ReflectRefract<scalar_type> rr;
        rr.refract = r;
        L.direction = rr(transmitted, orientedEta.rcp[0]);
        return sample_type::create(L, T, B, N);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        fresnel::OrientedEtas<monochrome_type> orientedEta = fresnel::OrientedEtas<monochrome_type>::create(interaction.getNdotV(_clamp), hlsl::promote<monochrome_type>(eta));
        fresnel::OrientedEtaRcps<monochrome_type> rcpEta = fresnel::OrientedEtaRcps<monochrome_type>::create(interaction.getNdotV(_clamp), hlsl::promote<monochrome_type>(eta));
        bool dummy;
        return __generate_wo_clamps(interaction.getV().getDirection(), interaction.getT(), interaction.getB(), interaction.getN(), interaction.getNdotV(), 
            interaction.getNdotV(_clamp), u, orientedEta, rcpEta, dummy);
    }
    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        return generate(anisotropic_interaction_type::create(interaction), u);
    }

    // eval and pdf return 0 because smooth dielectric/conductor BxDFs are dirac delta distributions, model perfectly specular objects that scatter light to only one outgoing direction
    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample)
    {
        return 0;
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        const bool transmitted = ComputeMicrofacetNormal<scalar_type>::isTransmissionPath(interaction.getNdotV(), _sample.getNdotL());

        fresnel::OrientedEtaRcps<monochrome_type> rcpOrientedEtas = fresnel::OrientedEtaRcps<monochrome_type>::create(interaction.getNdotV(_clamp), hlsl::promote<monochrome_type>(eta));

        const scalar_type _pdf = bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
        scalar_type quo = hlsl::mix<scalar_type, bool>(1.0, rcpOrientedEtas.value[0], transmitted);
        return quotient_pdf_type::create(quo, _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return quotient_and_pdf(_sample, interaction.isotropic);
    }

    scalar_type eta;
};

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SSmoothThinDielectricBxDF
{
    using this_t = SSmoothThinDielectricBxDF<Config>;
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

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_ABS;

    static this_t create(NBL_CONST_REF_ARG(spectral_type) eta2, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint)
    {
        this_t retval;
        retval.eta2 = eta2;
        retval.luminosityContributionHint = luminosityContributionHint;
        return retval;
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        return hlsl::promote<spectral_type>(0);
    }
    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return hlsl::promote<spectral_type>(0);
    }

    // usually `luminosityContributionHint` would be the Rec.709 luma coefficients (the Y row of the RGB to CIE XYZ matrix)
    // its basically a set of weights that determine
    // assert(1.0==luminosityContributionHint.r+luminosityContributionHint.g+luminosityContributionHint.b);
    // `remainderMetadata` is a variable which the generator function returns byproducts of sample generation that would otherwise have to be redundantly calculated `quotient_and_pdf`
    sample_type __generate_wo_clamps(const vector3_type V, const vector3_type T, const vector3_type B, const vector3_type N, scalar_type NdotV, scalar_type absNdotV, NBL_REF_ARG(vector3_type) u, NBL_CONST_REF_ARG(spectral_type) eta2, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint, NBL_REF_ARG(spectral_type) remainderMetadata)
    {
        // we will only ever intersect from the outside
        const spectral_type reflectance = fresnel::thinDielectricInfiniteScatter<spectral_type>(fresnel::Dielectric<spectral_type>::__call(eta2,absNdotV));

        // we are only allowed one choice for the entire ray, so make the probability a weighted sum
        const scalar_type reflectionProb = nbl::hlsl::dot<spectral_type>(reflectance, luminosityContributionHint);

        scalar_type rcpChoiceProb;
        const bool transmitted = math::partitionRandVariable(reflectionProb, u.z, rcpChoiceProb);
        remainderMetadata = (transmitted ? (hlsl::promote<spectral_type>(1.0) - reflectance) : reflectance) * rcpChoiceProb;

        ray_dir_info_type L;
        L.direction = (transmitted ? (vector3_type)(0.0) : N * 2.0f * NdotV) - V;
        return sample_type::create(L, T, B, N);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        vector3_type dummy;
        return __generate_wo_clamps(interaction.getV().getDirection(), interaction.getT(), interaction.getB(), interaction.getN(), interaction.getNdotV(), interaction.getNdotV(_clamp), u, eta2, luminosityContributionHint, dummy);
    }
    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        return generate(anisotropic_interaction_type::create(interaction), u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample)
    {
        return 0;
    }

    // isotropic only?
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        const bool transmitted = ComputeMicrofacetNormal<scalar_type>::isTransmissionPath(interaction.getNdotV(), _sample.getNdotL());
        const spectral_type reflectance = fresnel::thinDielectricInfiniteScatter<spectral_type>(fresnel::Dielectric<spectral_type>::__call(eta2, interaction.getNdotV(_clamp)));
        const spectral_type sampleValue = hlsl::mix(reflectance, hlsl::promote<spectral_type>(1.0) - reflectance, transmitted);

        const scalar_type sampleProb = nbl::hlsl::dot<spectral_type>(sampleValue,luminosityContributionHint);

        const scalar_type _pdf = bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
        return quotient_pdf_type::create(sampleValue / sampleProb, _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return quotient_and_pdf(_sample, interaction.isotropic);
    }

    spectral_type eta2;
    spectral_type luminosityContributionHint;
};

}

template<typename C>
struct traits<bxdf::transmission::SSmoothDielectricBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename C>
struct traits<bxdf::transmission::SSmoothThinDielectricBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
