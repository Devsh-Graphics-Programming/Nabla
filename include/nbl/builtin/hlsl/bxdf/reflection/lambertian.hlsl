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

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::BasicConfiguration<Config>)
struct SLambertian
{
    using this_t = SLambertian<Config>;
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

    scalar_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        return _sample.getNdotL(_clamp) * numbers::inv_pi<scalar_type>;
    }
    scalar_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return eval(_sample, interaction.isotropic);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector2_type u)
    {
        ray_dir_info_type L;
        L.direction = sampling::ProjectedHemisphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u)
    {
        return generate(anisotropic_interaction_type::create(interaction), u);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample)
    {
        return sampling::ProjectedHemisphere<scalar_type>::pdf(_sample.getNdotL(_clamp));
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        sampling::quotient_and_pdf<monochrome_type, scalar_type> qp = sampling::ProjectedHemisphere<scalar_type>::template quotient_and_pdf(_sample.getNdotL(_clamp));
        return quotient_pdf_type::create(qp.quotient[0], qp.pdf);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction)
    {
        return quotient_and_pdf(_sample, interaction.isotropic);
    }
};

}

template<typename C>
struct traits<bxdf::reflection::SLambertian<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
