// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_OREN_NAYAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_OREN_NAYAR_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/bxdf/base/oren_nayar.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SOrenNayar
{
    using this_t = SOrenNayar<Config>;
    BXDF_CONFIG_TYPE_ALIASES(Config);

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = BxDFClampMode::BCM_ABS;

    using base_type = base::SOrenNayarBase<Config, false>;
    using creation_type = typename base_type::creation_type;

    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        this_t retval;
        retval.__base = base_type::create(params);
        return retval;
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        return __base.eval(_sample, interaction);
    }
    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return __base.eval(_sample, interaction.isotropic);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector3_type u)
    {
        return __base.generate(anisotropic_interaction_type::create(interaction), u);
    }
    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type u)
    {
        return __base.generate(interaction, u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample)
    {
        return __base.pdf(_sample);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        return __base.quotient_and_pdf(_sample, interaction);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return __base.quotient_and_pdf(_sample, interaction.isotropic);
    }

    base_type __base;
};

}

template<typename C>
struct traits<bxdf::transmission::SOrenNayar<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
