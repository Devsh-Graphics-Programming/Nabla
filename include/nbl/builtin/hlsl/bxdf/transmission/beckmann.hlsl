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
    MICROFACET_BXDF_CONFIG_TYPE_ALIASES(Config);

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
        retval.__base.ndf.__ndf_base.a2 = A*A;
        retval.__base.ndf.__generate_base.ax = A;
        retval.__base.ndf.__generate_base.ay = A;
        retval.__base.fresnel.orientedEta = orientedEta;
        retval.__base.fresnel.orientedEta2 = orientedEta.value * orientedEta.value;
        retval.__base.luminosityContributionHint = luminosityContributionHint;
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
    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __base.eval(_sample, interaction.isotropic, cache.iso_cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector3_type u, NBL_REF_ARG(isocache_type) cache)
    {
        anisocache_type aniso_cache;
        sample_type s = __base.template generate<vector3_type>(anisotropic_interaction_type::create(interaction), u, aniso_cache);
        cache = aniso_cache.iso_cache;
        return s;
    }
    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        return __base.template generate<vector3_type>(interaction, u, cache);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __base.pdf(_sample, interaction, cache);      
    }
    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __base.pdf(_sample, interaction.isotropic, cache.iso_cache);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __base.quotient_and_pdf(_sample, interaction, cache);
    }
    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __base.quotient_and_pdf(_sample, interaction.isotropic, cache.iso_cache);
    }

    SCookTorrance<Config, ndf_type, fresnel_type> __base;
};

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannDielectricAnisotropic
{
    using this_t = SBeckmannDielectricAnisotropic<Config>;
    MICROFACET_BXDF_CONFIG_TYPE_ALIASES(Config);

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
        retval.__base.ndf.__ndf_base.ax2 = ax*ax;
        retval.__base.ndf.__ndf_base.ay2 = ay*ay;
        retval.__base.ndf.__ndf_base.a2 = ax*ay;
        retval.__base.ndf.__generate_base.ax = ax;
        retval.__base.ndf.__generate_base.ay = ay;
        retval.__base.fresnel.orientedEta = orientedEta;
        retval.__base.fresnel.orientedEta2 = orientedEta.value * orientedEta.value;
        retval.__base.luminosityContributionHint = luminosityContributionHint;
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

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        return __base.template generate<vector3_type>(interaction, u, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector3_type u)
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

    SCookTorrance<Config, ndf_type, fresnel_type> __base;
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
