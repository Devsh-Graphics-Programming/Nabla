// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_OREN_NAYAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_OREN_NAYAR_INCLUDED_

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

template<class LS, class SI, typename Scalar NBL_STRUCT_CONSTRAINABLE>
struct OrenNayarParams;

template<class LS, class SI, typename Scalar>
NBL_PARTIAL_REQ_TOP(!surface_interactions::Anisotropic<SI>)
struct OrenNayarParams<LS, SI, Scalar NBL_PARTIAL_REQ_BOT(!surface_interactions::Anisotropic<SI>) >
{
    using this_t = OrenNayarParams<LS, SI, Scalar>;

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
struct OrenNayarParams<LS, SI, Scalar NBL_PARTIAL_REQ_BOT(surface_interactions::Anisotropic<SI>) >
{
    using this_t = OrenNayarParams<LS, SI, Scalar>;

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
struct SOrenNayarBxDF
{
    using this_t = SOrenNayarBxDF<LS, Iso, Aniso, Spectrum>;
    using scalar_type = typename LS::scalar_type;
    using vector2_type = vector<scalar_type, 2>;
    using ray_dir_info_type = typename LS::ray_dir_info_type;

    using isotropic_interaction_type = Iso;
    using anisotropic_interaction_type = Aniso;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    
    using params_isotropic_t = OrenNayarParams<LS, Iso, scalar_type>;
    using params_anisotropic_t = OrenNayarParams<LS, Aniso, scalar_type>;


    static this_t create(scalar_type A)
    {
        this_t retval;
        retval.A = A;
        return retval;
    }

    static this_t create(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        return create(params.A.x);
    }

    void init(NBL_CONST_REF_ARG(SBxDFCreationParams<scalar_type, spectral_type>) params)
    {
        A = params.A.x;
    }

    scalar_type __rec_pi_factored_out_wo_clamps(scalar_type VdotL, scalar_type maxNdotL, scalar_type maxNdotV)
    {
        scalar_type A2 = A * 0.5;
        vector2_type AB = vector2_type(1.0, 0.0) + vector2_type(-0.5, 0.45) * vector2_type(A2, A2) / vector2_type(A2 + 0.33, A2 + 0.09);
        scalar_type C = 1.0 / max<scalar_type>(maxNdotL, maxNdotV);

        scalar_type cos_phi_sin_theta = max<scalar_type>(VdotL - maxNdotL * maxNdotV, 0.0);
        return (AB.x + AB.y * cos_phi_sin_theta * C);
    }

    scalar_type eval(NBL_CONST_REF_ARG(params_isotropic_t) params)
    {
        return params.getNdotL() * numbers::inv_pi<scalar_type> * __rec_pi_factored_out_wo_clamps(params.getVdotL(), params.getNdotL(), params.getNdotV());
    }
    scalar_type eval(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        return params.getNdotL() * numbers::inv_pi<scalar_type> * __rec_pi_factored_out_wo_clamps(params.getVdotL(), params.getNdotL(), params.getNdotV());
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector2_type) u)
    {
        ray_dir_info_type L;
        L.direction = sampling::ProjectedHemisphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(interaction.getTangentSpaceV(), L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector2_type) u)
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
        scalar_type _pdf = pdf(params);
        scalar_type q = __rec_pi_factored_out_wo_clamps(params.getVdotL(), params.getNdotL(), params.getNdotV());
        return quotient_pdf_type::create(hlsl::promote<spectral_type>(q), _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(params_anisotropic_t) params)
    {
        scalar_type _pdf = pdf(params);
        scalar_type q = __rec_pi_factored_out_wo_clamps(params.getVdotL(), params.getNdotL(), params.getNdotV());
        return quotient_pdf_type::create(hlsl::promote<spectral_type>(q), _pdf);
    }

    scalar_type A;
};

}
}
}
}

#endif
