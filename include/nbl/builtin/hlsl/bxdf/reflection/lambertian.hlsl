// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_LAMBERTIAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_LAMBERTIAN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
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
    Scalar getNdotVUnclamped() NBL_CONST_MEMBER_FUNC { return interaction.getNdotV(); }
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
    using scalar_type = typename Config::scalar_type;
    using ray_dir_info_type = typename Config::ray_dir_info_type;
    using isotropic_interaction_type = typename Config::isotropic_interaction_type;
    using anisotropic_interaction_type = typename Config::anisotropic_interaction_type;
    using sample_type = typename Config::sample_type;
    using spectral_type = typename Config::spectral_type;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;

    using params_isotropic_t = LambertianParams<sample_type, isotropic_interaction_type, scalar_type>;
    using params_anisotropic_t = LambertianParams<sample_type, anisotropic_interaction_type, scalar_type>;


    static this_t create()
    {
        this_t retval;
        // nothing here, just keeping in convention with others
        return retval;
    }

    scalar_type __eval_pi_factored_out(scalar_type maxNdotL)
    {
        return maxNdotL;
    }

    scalar_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return __eval_pi_factored_out(params.getNdotL()) * numbers::inv_pi<scalar_type>;
    }
    scalar_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return __eval_pi_factored_out(params.getNdotL()) * numbers::inv_pi<scalar_type>;
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 2>) u)
    {
        ray_dir_info_type L;
        L.direction = sampling::ProjectedHemisphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(interaction.getTangentSpaceV(), L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 2>) u)
    {
        return generate_wo_clamps(interaction, u);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 2>) u)
    {
        return generate_wo_clamps(anisotropic_interaction_type::create(interaction), u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return sampling::ProjectedHemisphere<scalar_type>::pdf(params.getNdotL());
    }
    scalar_type pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return sampling::ProjectedHemisphere<scalar_type>::pdf(params.getNdotL());
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        sampling::quotient_and_pdf<vector<scalar_type,1>, scalar_type> qp = sampling::ProjectedHemisphere<scalar_type>::template quotient_and_pdf(params.getNdotL());
        return quotient_pdf_type::create(hlsl::promote<spectral_type>(qp.quotient[0]), qp.pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        sampling::quotient_and_pdf<vector<scalar_type,1>, scalar_type> qp = sampling::ProjectedHemisphere<scalar_type>::template quotient_and_pdf(params.getNdotL());
        return quotient_pdf_type::create(hlsl::promote<spectral_type>(qp.quotient[0]), qp.pdf);
    }
};

}

template<typename C>
struct traits<bxdf::reflection::SLambertianBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
