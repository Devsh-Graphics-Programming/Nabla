// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_BASE_LAMBERTIAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BASE_LAMBERTIAN_INCLUDED_

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
struct SLambertianBase
{
    using this_t = SLambertianBase<Config, IsBSDF>;
    BXDF_CONFIG_TYPE_ALIASES(Config);

    using random_type = conditional_t<IsBSDF, vector3_type, vector2_type>;
    struct Cache {};
    using isocache_type = Cache;
    using anisocache_type = Cache;

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = conditional_value<IsBSDF, BxDFClampMode, BxDFClampMode::BCM_ABS, BxDFClampMode::BCM_MAX>::value;

    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        const spectral_type quo = hlsl::promote<spectral_type>(_sample.getNdotL(_clamp) * numbers::inv_pi<scalar_type> * hlsl::mix(1.0, 0.5, IsBSDF));
        return value_weight_type::create(quo, forwardPdf(_sample, interaction));
    }
    value_weight_type evalAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return evalAndWeight(_sample, interaction.isotropic);
    }

    template<typename C=bool_constant<!IsBSDF> >
    enable_if_t<C::value && !IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const random_type u, NBL_CONST_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        typename sampling::ProjectedHemisphere<scalar_type>::cache_type cache;
        ray_dir_info_type L;
        L.setDirection(sampling::ProjectedHemisphere<scalar_type>::generate(u, cache));
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }
    template<typename C=bool_constant<IsBSDF> >
    enable_if_t<C::value && IsBSDF, sample_type> generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const random_type u, NBL_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        typename sampling::ProjectedSphere<scalar_type>::cache_type cache;
        vector3_type _u = u;
        ray_dir_info_type L;
        L.setDirection(sampling::ProjectedSphere<scalar_type>::generate(_u, cache));
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }
    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const random_type u, NBL_REF_ARG(isocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        return generate(anisotropic_interaction_type::create(interaction), u, _cache);
    }

    scalar_type forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        NBL_IF_CONSTEXPR (IsBSDF)
            return sampling::ProjectedSphere<scalar_type>::pdf(_sample.getNdotL(_clamp));
        else
            return sampling::ProjectedHemisphere<scalar_type>::pdf(_sample.getNdotL(_clamp));
    }
    scalar_type forwardPdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction) NBL_CONST_MEMBER_FUNC
    {
        return forwardPdf(_sample, interaction.isotropic);
    }

    quotient_weight_type quotientAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        sampling::quotient_and_pdf<monochrome_type, scalar_type> qp;
        NBL_IF_CONSTEXPR (IsBSDF)
            qp = sampling::ProjectedSphere<scalar_type>::template quotientAndPdf(_sample.getNdotL(_clamp));
        else
            qp = sampling::ProjectedHemisphere<scalar_type>::template quotientAndPdf(_sample.getNdotL(_clamp));
        return quotient_weight_type::create(qp.quotient()[0], qp.pdf());
    }
    quotient_weight_type quotientAndWeight(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) _cache) NBL_CONST_MEMBER_FUNC
    {
        return quotientAndWeight(_sample, interaction.isotropic, _cache);
    }
};

}
}
}
}

#endif
