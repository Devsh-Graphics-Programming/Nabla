// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_BECKMANN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/cook_torrance_base.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannDielectricIsotropic
{
    using this_t = SBeckmannDielectricIsotropic<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(matrix3x3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(monochrome_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);

    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);

    using ndf_type = ndf::Beckmann<scalar_type, false, ndf::MTT_REFLECT_REFRACT>;
    using fresnel_type = fresnel::Dielectric<monochrome_type>;

    struct SCreationParams
    {
        scalar_type A;
        fresnel::OrientedEtas<monochrome_type> orientedEta;
        spectral_type luminosityContributionHint;
    };
    using creation_type = SCreationParams;

    static this_t create(NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEta, scalar_type A, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint)
    {
        this_t retval;
        retval.__base.ndf.__base.A = vector2_type(A, A);
        retval.__base.ndf.__base.a2 = A*A;
        retval.__base.fresnel.orientedEta = orientedEta;
        retval.__base.fresnel.orientedEta2 = orientedEta.value * orientedEta.value;
        retval.luminosityContributionHint = luminosityContributionHint;
        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEta, scalar_type A)
    {
        static_assert(vector_traits<spectral_type>::Dimension == 3);
        const spectral_type rec709 = spectral_type(0.2126, 0.7152, 0.0722);
        return create(orientedEta, A, rec709);
    }
    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        return create(params.orientedEta, params.A, params.luminosityContributionHint);
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __base.eval(_sample, interaction, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(isocache_type) cache)
    {
        return __base.generate(interaction, u, cache, luminosityContributionHint);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __base.pdf(_sample, interaction, cache);      
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __base.quotient_and_pdf(_sample, interaction, cache);
    }

    SCookTorrance<Config, ndf_type, fresnel_type, true> __base;
    spectral_type luminosityContributionHint;
};

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannDielectricAnisotropic
{
    using this_t = SBeckmannDielectricAnisotropic<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(matrix3x3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(monochrome_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);

    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);

    using ndf_type = ndf::Beckmann<scalar_type, true, ndf::MTT_REFLECT_REFRACT>;
    using fresnel_type = fresnel::Dielectric<monochrome_type>;

    struct SCreationParams
    {
        scalar_type ax;
        scalar_type ay;
        fresnel::OrientedEtas<monochrome_type> orientedEta;
        spectral_type luminosityContributionHint;
    };
    using creation_type = SCreationParams;

    static this_t create(NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEta, scalar_type ax, scalar_type ay, NBL_CONST_REF_ARG(spectral_type) luminosityContributionHint)
    {
        this_t retval;
        retval.__base.ndf.__base.A = vector2_type(ax, ay);
        retval.__base.ndf.__base.ax2 = ax*ax;
        retval.__base.ndf.__base.ay2 = ay*ay;
        retval.__base.ndf.__base.a2 = ax*ay;
        retval.__base.fresnel.orientedEta = orientedEta;
        retval.__base.fresnel.orientedEta2 = orientedEta.value * orientedEta.value;
        retval.luminosityContributionHint = luminosityContributionHint;
        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(fresnel::OrientedEtas<monochrome_type>) orientedEta, scalar_type ax, scalar_type ay)
    {
        static_assert(vector_traits<spectral_type>::Dimension == 3);
        const spectral_type rec709 = spectral_type(0.2126, 0.7152, 0.0722);
        return create(orientedEta, ax, ay, rec709);
    }
    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        return create(params.orientedEta, params.ax, params.ay, params.luminosityContributionHint);
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __base.eval(_sample, interaction, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u, NBL_REF_ARG(anisocache_type) cache)
    {
        return __base.generate(interaction, u, cache, luminosityContributionHint);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_REF_ARG(vector3_type) u)
    {
        anisocache_type dummycache;
        return generate(interaction, u, dummycache);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __base.pdf(_sample, interaction, cache);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __base.quotient_and_pdf(_sample, interaction, cache);
    }

    SCookTorrance<Config, ndf_type, fresnel_type, true> __base;
    spectral_type luminosityContributionHint;
};

}

template<typename C>
struct traits<bxdf::transmission::SBeckmannDielectricIsotropic<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename C>
struct traits<bxdf::transmission::SBeckmannDielectricAnisotropic<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
