// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_BASE_OREN_NAYAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BASE_OREN_NAYAR_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace base
{

template<class Config, bool IsBSDF NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SOrenNayarBase
{
    using this_t = SOrenNayarBase<Config, IsBSDF>;
    BXDF_CONFIG_TYPE_ALIASES(Config);

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = conditional_value<IsBSDF, BxDFClampMode, BxDFClampMode::BCM_ABS, BxDFClampMode::BCM_MAX>::value;

    struct SCreationParams
    {
        scalar_type A;
    };
    using creation_type = SCreationParams;

    struct SQuery
    {
        scalar_type getVdotL() NBL_CONST_MEMBER_FUNC { return VdotL; }
        scalar_type VdotL;
    };

    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        this_t retval;
        retval.A2 = params.A * 0.5;
        retval.AB = vector2_type(1.0, 0.0) + vector2_type(-0.5, 0.45) * vector2_type(retval.A2, retval.A2) / vector2_type(retval.A2 + 0.33, retval.A2 + 0.09);
        return retval;
    }

    scalar_type __rec_pi_factored_out_wo_clamps(scalar_type VdotL, scalar_type clampedNdotL, scalar_type clampedNdotV) NBL_CONST_MEMBER_FUNC
    {
        scalar_type C = 1.0 / max<scalar_type>(clampedNdotL, clampedNdotV);
        scalar_type cos_phi_sin_theta = max<scalar_type>(VdotL - clampedNdotL * clampedNdotV, 0.0);
        return (AB.x + AB.y * cos_phi_sin_theta * C);
    }
    template<typename Query>
    spectral_type __eval(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        scalar_type NdotL = _sample.getNdotL(_clamp);
        return hlsl::promote<spectral_type>(NdotL * numbers::inv_pi<scalar_type> * hlsl::mix(1.0, 0.5, IsBSDF) * __rec_pi_factored_out_wo_clamps(query.getVdotL(), NdotL, interaction.getNdotV(_clamp)));
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        SQuery query;
        query.VdotL = hlsl::dot(interaction.getV().getDirection(), _sample.getL().getDirection());
        return __eval<SQuery>(query, _sample, interaction); 
    }
    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return eval(_sample, interaction.isotropic); 
    }

    template<typename C=bool_constant<!IsBSDF> >
    enable_if_t<C::value && !IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector2_type u) NBL_CONST_MEMBER_FUNC
    {
        ray_dir_info_type L;
        L.setDirection(sampling::ProjectedHemisphere<scalar_type>::generate(u));
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }
    template<typename C=bool_constant<IsBSDF> >
    enable_if_t<C::value && IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type u) NBL_CONST_MEMBER_FUNC
    {
        vector3_type _u = u;
        ray_dir_info_type L;
        L.setDirection(sampling::ProjectedSphere<scalar_type>::generate(_u));
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }
    template<typename C=bool_constant<!IsBSDF> >
    enable_if_t<C::value && !IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u) NBL_CONST_MEMBER_FUNC
    {
        return generate(anisotropic_interaction_type::create(interaction), u);
    }
    template<typename C=bool_constant<IsBSDF> >
    enable_if_t<C::value && IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector3_type u) NBL_CONST_MEMBER_FUNC
    {
        return generate(anisotropic_interaction_type::create(interaction), u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        if (IsBSDF)
            return sampling::ProjectedSphere<scalar_type>::pdf(_sample.getNdotL(_clamp));
        else
            return sampling::ProjectedHemisphere<scalar_type>::pdf(_sample.getNdotL(_clamp));
    }
    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return pdf(_sample, interaction.isotropic);
    }

    template<typename Query>
    quotient_pdf_type __quotient_and_pdf(NBL_CONST_REF_ARG(Query) query, NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        scalar_type _pdf = pdf(_sample, interaction);
        scalar_type q = __rec_pi_factored_out_wo_clamps(query.getVdotL(), _sample.getNdotL(_clamp), interaction.getNdotV(_clamp));
        return quotient_pdf_type::create(q, _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        SQuery query;
        query.VdotL = hlsl::dot(interaction.getV().getDirection(), _sample.getL().getDirection());
        return __quotient_and_pdf<SQuery>(query, _sample, interaction);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return quotient_and_pdf(_sample, interaction.isotropic);
    }

    scalar_type A2;
    vector2_type AB;
};

}
}
}
}

#endif
