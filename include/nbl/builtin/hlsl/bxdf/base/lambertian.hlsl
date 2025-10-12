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

    NBL_CONSTEXPR_STATIC_INLINE BxDFClampMode _clamp = conditional_value<IsBSDF, BxDFClampMode, BxDFClampMode::BCM_ABS, BxDFClampMode::BCM_MAX>::value;

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        return hlsl::promote<spectral_type>(_sample.getNdotL(_clamp) * numbers::inv_pi<scalar_type> * hlsl::mix(1.0, 0.5, IsBSDF));
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector2_type u)
    {
        // static_assert(!IsBSDF);
        ray_dir_info_type L;
        L.direction = sampling::ProjectedHemisphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type u)
    {
        // static_assert(IsBSDF);
        ray_dir_info_type L;
        L.direction = sampling::ProjectedSphere<scalar_type>::generate(u);
        return sample_type::createFromTangentSpace(L, interaction.getFromTangentSpace());
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample)
    {
        if (IsBSDF)
            return sampling::ProjectedSphere<scalar_type>::pdf(_sample.getNdotL(_clamp));
        else
            return sampling::ProjectedHemisphere<scalar_type>::pdf(_sample.getNdotL(_clamp));
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction)
    {
        sampling::quotient_and_pdf<monochrome_type, scalar_type> qp;
        if (IsBSDF)
            qp = sampling::ProjectedSphere<scalar_type>::template quotient_and_pdf(_sample.getNdotL(_clamp));
        else
            qp = sampling::ProjectedHemisphere<scalar_type>::template quotient_and_pdf(_sample.getNdotL(_clamp));
        return quotient_pdf_type::create(qp.quotient[0], qp.pdf);
    }
};

}
}
}
}

#endif
