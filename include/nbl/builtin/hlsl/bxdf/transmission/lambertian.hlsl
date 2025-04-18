// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_LAMBERTIAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_LAMBERTIAN_INCLUDED_

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

template<class LS, class Iso, class Aniso, class Spectrum NBL_PRIMARY_REQUIRES(LightSample<LS> && surface_interactions::Isotropic<Iso> && surface_interactions::Anisotropic<Aniso>)
struct SLambertianBxDF
{
    using this_t = SLambertianBxDF<LS, Iso, Aniso, Spectrum>;
    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using isotropic_interaction_type = Iso;
    using anisotropic_interaction_type = Aniso;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;

    using params_isotropic_t = LambertianParams<LS, Iso, scalar_type>;
    using params_anisotropic_t = LambertianParams<LS, Aniso, scalar_type>;


    static this_t create()
    {
        this_t retval;
        // nothing here, just keeping convention with others
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        return create();
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        // do nothing
    }

    scalar_type __eval_pi_factored_out(scalar_type absNdotL)
    {
        return absNdotL;
    }

    scalar_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return __eval_pi_factored_out(params.getNdotL()) * numbers::inv_pi<scalar_type> * 0.5;
    }
    scalar_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return __eval_pi_factored_out(params.getNdotL()) * numbers::inv_pi<scalar_type> * 0.5;
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 3>) u)
    {
        ray_dir_info_type L;
        L.direction = sampling::ProjectedSphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(interaction.getTangentSpaceV(), L, interaction.getFromTangentSpace());
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
        sampling::quotient_and_pdf<scalar_type, scalar_type> qp = sampling::ProjectedSphere<scalar_type>::template quotient_and_pdf(params.getNdotL());
        return quotient_pdf_type::create(hlsl::promote<spectral_type>(qp.quotient), qp.pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        sampling::quotient_and_pdf<scalar_type, scalar_type> qp = sampling::ProjectedSphere<scalar_type>::template quotient_and_pdf(params.getNdotL());
        return quotient_pdf_type::create(hlsl::promote<spectral_type>(qp.quotient), qp.pdf);
    }
};

}
}
}
}

#endif
