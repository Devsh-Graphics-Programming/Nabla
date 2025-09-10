// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_BECKMANN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/beckmann.hlsl"
#include "nbl/builtin/hlsl/bxdf/cook_torrance_base.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

// template<class Config NBL_STRUCT_CONSTRAINABLE>
// struct SBeckmannAnisotropic;

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannIsotropic
{
    using this_t = SBeckmannIsotropic<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);

    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);

    using ndf_type = ndf::Beckmann<scalar_type, false, ndf::MTT_REFLECT>;
    using fresnel_type = fresnel::Conductor<spectral_type>;

    struct SCreationParams
    {
        scalar_type A;
        spectral_type ior0;
        spectral_type ior1;
    };
    using creation_type = SCreationParams;

    static this_t create(scalar_type A, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.__base.ndf.__base.A = vector2_type(A, A);
        retval.__base.ndf.__base.a2 = A*A;
        retval.__base.fresnel.ior = ior0;
        retval.__base.fresnel.iork = ior1;
        retval.__base.fresnel.iork2 = ior1*ior1;
        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        return create(params.A, params.ior0, params.ior1);
    }

    // TODO: isotropic should also include anisotropic overloads

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __base.eval(_sample, interaction, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(isocache_type) cache)
    {
        return __base.generate(interaction, u, cache);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __base.pdf(_sample, interaction, cache);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(isotropic_interaction_type) interaction, NBL_CONST_REF_ARG(isocache_type) cache)
    {
        return __base.quotient_and_pdf(_sample, interaction, cache);
    }

    SCookTorrance<Config, ndf_type, fresnel_type, false> __base;
};

template<class Config NBL_PRIMARY_REQUIRES(config_concepts::MicrofacetConfiguration<Config>)
struct SBeckmannAnisotropic
{
    using this_t = SBeckmannAnisotropic<Config>;
    NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);
    NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);
    NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);

    NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);
    NBL_BXDF_CONFIG_ALIAS(sample_type, Config);
    NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);
    NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);
    NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);
    NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);

    using ndf_type = ndf::Beckmann<scalar_type, true, ndf::MTT_REFLECT>;
    using fresnel_type = fresnel::Conductor<spectral_type>;

    struct SCreationParams
    {
        scalar_type ax;
        scalar_type ay;
        spectral_type ior0;
        spectral_type ior1;
    };
    using creation_type = SCreationParams;

    static this_t create(scalar_type ax, scalar_type ay, NBL_CONST_REF_ARG(spectral_type) ior0, NBL_CONST_REF_ARG(spectral_type) ior1)
    {
        this_t retval;
        retval.__base.ndf.__base.A = vector2_type(ax, ay);
        retval.__base.ndf.__base.ax2 = ax*ax;
        retval.__base.ndf.__base.ay2 = ay*ay;
        retval.__base.ndf.__base.a2 = ax*ay;
        retval.__base.fresnel.ior = ior0;
        retval.__base.fresnel.iork = ior1;
        retval.__base.fresnel.iork2 = ior1*ior1;
        return retval;
    }
    static this_t create(NBL_CONST_REF_ARG(creation_type) params)
    {
        return create(params.ax, params.ay, params.ior0, params.ior1);
    }

    spectral_type eval(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __base.eval(_sample, interaction, cache);
    }

    sample_type generate(NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, const vector2_type u, NBL_REF_ARG(anisocache_type) cache)
    {
        return __base.generate(interaction, u, cache);
    }

    scalar_type pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __base.pdf(_sample, interaction, cache);
    }

    quotient_pdf_type quotient_and_pdf(NBL_CONST_REF_ARG(sample_type) _sample, NBL_CONST_REF_ARG(anisotropic_interaction_type) interaction, NBL_CONST_REF_ARG(anisocache_type) cache)
    {
        return __base.quotient_and_pdf(_sample, interaction, cache);
    }

    SCookTorrance<Config, ndf_type, fresnel_type, false> __base;
};

}

template<typename C>
struct traits<bxdf::reflection::SBeckmannIsotropic<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename C>
struct traits<bxdf::reflection::SBeckmannAnisotropic<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
