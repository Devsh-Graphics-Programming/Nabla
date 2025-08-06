// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_OREN_NAYAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_OREN_NAYAR_INCLUDED_

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

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SOrenNayarBxDF
{
    using this_t = SOrenNayarBxDF<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_MAX;

    static this_t create(scalar_type A)
    {
        this_t retval;
        retval.A = A;
        return retval;
    }

    scalar_type __rec_pi_factored_out_wo_clamps(scalar_type VdotL, scalar_type maxNdotL, scalar_type maxNdotV)
    {
        scalar_type A2 = A * 0.5;
        vector2_type AB = vector2_type(1.0, 0.0) + vector2_type(-0.5, 0.45) * vector2_type(A2, A2) / vector2_type(A2 + 0.33, A2 + 0.09);
        scalar_type C = 1.0 / max<scalar_type>(maxNdotL, maxNdotV);

        scalar_type cos_phi_sin_theta = max<scalar_type>(VdotL - maxNdotL * maxNdotV, 0.0);
        return (AB.x + AB.y * cos_phi_sin_theta * C);
    }

    scalar_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        scalar_type NdotL = _sample.getNdotL(_clamp);
        return NdotL * numbers::inv_pi<scalar_type> * __rec_pi_factored_out_wo_clamps(hlsl::dot(interaction.getV().getDirection(), _sample.getL().getDirection()), NdotL, interaction.getNdotV(_clamp));
    }
    scalar_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return eval(_sample, interaction.isotropic);
    }

    sample_type generate_wo_clamps(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector2_type) u)
    {
        ray_dir_info_type L;
        L.direction = sampling::ProjectedHemisphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector2_type) u)
    {
        return generate_wo_clamps(interaction, u);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(vector<scalar_type, 2>) u)
    {
        return generate_wo_clamps(anisotropic_interaction_type::create(interaction), u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample)
    {
        return sampling::ProjectedHemisphere<scalar_type>::pdf(_sample.getNdotL(_clamp));
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        scalar_type _pdf = pdf(_sample);
        scalar_type q = __rec_pi_factored_out_wo_clamps(hlsl::dot(interaction.getV().getDirection(), _sample.getL().getDirection()), _sample.getNdotL(_clamp), interaction.getNdotV(_clamp));
        return quotient_pdf_type::create(q, _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return quotient_and_pdf(_sample, interaction.isotropic);
    }

    scalar_type A;
};

}

template<typename C>
struct traits<bxdf::reflection::SOrenNayarBxDF<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
