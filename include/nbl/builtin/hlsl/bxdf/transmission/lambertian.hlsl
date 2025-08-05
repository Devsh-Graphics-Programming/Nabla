// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_LAMBERTIAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_LAMBERTIAN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
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

template<class LS, class SI, typename Scalar NBL_STRUCT_CONSTRAINABLE>
struct LambertianParams;

template<class LS, class SI, typename Scalar>
NBL_PARTIAL_REQ_TOP(!surface_interactions::Anisotropic<SI>)
struct LambertianParams<LS, SI, Scalar NBL_PARTIAL_REQ_BOT(!surface_interactions::Anisotropic<SI>) >
{
    using this_t = LambertianParams<LS, SI, Scalar>;

    static this_t create(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(SI) interaction, BxDFClampMode _clamp)
    {
        this_t retval;
        retval._sample = _sample;
        retval.interaction = interaction;
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

    LS _sample;
    SI interaction;
    BxDFClampMode _clamp;
};
template<class LS, class SI, typename Scalar>
NBL_PARTIAL_REQ_TOP(surface_interactions::Anisotropic<SI>)
struct LambertianParams<LS, SI, Scalar NBL_PARTIAL_REQ_BOT(surface_interactions::Anisotropic<SI>) >
{
    using this_t = LambertianParams<LS, SI, Scalar>;

    static this_t create(NBL_CONST_REF_ARG(LS) _sample, NBL_CONST_REF_ARG(SI) interaction, BxDFClampMode _clamp)
    {
        this_t retval;
        retval._sample = _sample;
        retval.interaction = interaction;
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

    // aniso
    Scalar getTdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getTdotL2(); }
    Scalar getBdotL2() NBL_CONST_MEMBER_FUNC { return _sample.getBdotL2(); }
    Scalar getTdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getTdotV2(); }
    Scalar getBdotV2() NBL_CONST_MEMBER_FUNC { return interaction.getBdotV2(); }

    LS _sample;
    SI interaction;
    BxDFClampMode _clamp;
};

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SLambertianBxDF
{
    using this_t = SLambertianBxDF<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);

    using params_isotropic_t = LambertianParams<sample_type, isotropic_interaction_type, scalar_type>;
    using params_anisotropic_t = LambertianParams<sample_type, anisotropic_interaction_type, scalar_type>;


    static this_t create()
    {
        this_t retval;
        // nothing here, just keeping convention with others
        return retval;
    }

    scalar_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return params.getNdotL() * numbers::inv_pi<scalar_type> * 0.5;
    }
    scalar_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return params.getNdotL() * numbers::inv_pi<scalar_type> * 0.5;
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 3>) u)
    {
        ray_dir_info_type L;
        L.direction = sampling::ProjectedSphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 3>) u)
    {
        return generate_wo_clamps(interaction, u);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 3>) u)
    {
        return generate_wo_clamps(anisotropic_interaction_type::create(interaction), u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return sampling::ProjectedSphere<scalar_type>::pdf(params.getNdotL());
    }
    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return sampling::ProjectedSphere<scalar_type>::pdf(params.getNdotL());
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        sampling::quotient_and_pdf<vector<scalar_type,1>, scalar_type> qp = sampling::ProjectedSphere<scalar_type>::template quotient_and_pdf(params.getNdotL());
        return quotient_pdf_type::create(qp.quotient[0], qp.pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        sampling::quotient_and_pdf<vector<scalar_type,1>, scalar_type> qp = sampling::ProjectedSphere<scalar_type>::template quotient_and_pdf(params.getNdotL());
        return quotient_pdf_type::create(qp.quotient[0], qp.pdf);
    }
};

}

template<typename C>
struct traits<bxdf::transmission::SLambertianBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
