// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_DELTA_DISTRIBUTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_DELTA_DISTRIBUTION_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/config.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SDeltaDistribution
{
    using this_t = SDeltaDistribution<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(monochrome_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_MAX;

    struct SCreationParams {};
    using creation_type = SCreationParams;

    static this_t create()
    {
        this_t retval;
        // nothing here, just keeping in convention with others
        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        return create();
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        return hlsl::promote<spectral_type>(0);
    }
    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return hlsl::promote<spectral_type>(0);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector2_type u)
    {
        ray_dir_info_type L = interaction.getV().transmit();
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u)
    {
        return generate(anisotropic_interaction_type::create(interaction), u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample)
    {
        return 0;
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        const scalar_type _pdf = bit_cast<scalar_type, uint32_t>(numeric_limits<scalar_type>::infinity);
        return quotient_pdf_type::create(1.0, _pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return quotient_and_pdf(_sample, interaction.isotropic);
    }
};

}

template<typename C>
struct traits<bxdf::transmission::SDeltaDistribution<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
