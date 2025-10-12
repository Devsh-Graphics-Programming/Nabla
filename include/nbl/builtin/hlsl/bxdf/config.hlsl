// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_CONFIG_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_CONFIG_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{

// TODO should just check `is_base_of` (but not possible right now cause of DXC limitations)
namespace config_concepts
{
#define NBL_CONCEPT_NAME BasicConfiguration
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (conf, T)
NBL_CONCEPT_BEGIN(1)
#define conf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector2_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quotient_pdf_type))
);
#undef conf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME MicrofacetConfiguration
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (conf, T)
NBL_CONCEPT_BEGIN(1)
#define conf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(BasicConfiguration, T))
    ((NBL_CONCEPT_REQ_TYPE)(T::monochrome_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isocache_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisocache_type))
);
#undef conf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}

template<class LS, class Interaction, class Spectrum NBL_STRUCT_CONSTRAINABLE>
struct SConfiguration;

#define CONF_ISO LightSample<LS> && surface_interactions::Isotropic<Interaction> && !surface_interactions::Anisotropic<Interaction> && concepts::FloatingPointLikeVectorial<Spectrum>

template<class LS, class Interaction, class Spectrum>
NBL_PARTIAL_REQ_TOP(CONF_ISO)
struct SConfiguration<LS,Interaction,Spectrum NBL_PARTIAL_REQ_BOT(CONF_ISO) >
#undef CONF_ISO
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = false;

    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using monochrome_type = vector<scalar_type, 1>;

    using isotropic_interaction_type = Interaction;
    using anisotropic_interaction_type = surface_interactions::SAnisotropic<isotropic_interaction_type>;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
};

#define CONF_ANISO LightSample<LS> && surface_interactions::Anisotropic<Interaction> && concepts::FloatingPointLikeVectorial<Spectrum>

template<class LS, class Interaction, class Spectrum>
NBL_PARTIAL_REQ_TOP(CONF_ANISO)
struct SConfiguration<LS,Interaction,Spectrum NBL_PARTIAL_REQ_BOT(CONF_ANISO) >
#undef CONF_ANISO
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = true;

    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using monochrome_type = vector<scalar_type, 1>;

    using isotropic_interaction_type = typename Interaction::isotropic_interaction_type;
    using anisotropic_interaction_type = Interaction;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
};

template<class LS, class Interaction, class MicrofacetCache, class Spectrum NBL_STRUCT_CONSTRAINABLE>
struct SMicrofacetConfiguration;

#define MICROFACET_CONF_ISO LightSample<LS> && surface_interactions::Isotropic<Interaction> && !surface_interactions::Anisotropic<Interaction> && CreatableIsotropicMicrofacetCache<MicrofacetCache> && !AnisotropicMicrofacetCache<MicrofacetCache> && concepts::FloatingPointLikeVectorial<Spectrum>

template<class LS, class Interaction, class MicrofacetCache, class Spectrum>
NBL_PARTIAL_REQ_TOP(MICROFACET_CONF_ISO)
struct SMicrofacetConfiguration<LS,Interaction,MicrofacetCache,Spectrum NBL_PARTIAL_REQ_BOT(MICROFACET_CONF_ISO) > : SConfiguration<LS, Interaction, Spectrum>
#undef MICROFACET_CONF_ISO
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = false;

    using base_type = SConfiguration<LS, Interaction, Spectrum>;

    using matrix3x3_type = matrix<typename base_type::scalar_type,3,3>;

    using isocache_type = MicrofacetCache;
    using anisocache_type = SAnisotropicMicrofacetCache<MicrofacetCache>;
};

#define MICROFACET_CONF_ANISO LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache> && concepts::FloatingPointLikeVectorial<Spectrum>

template<class LS, class Interaction, class MicrofacetCache, class Spectrum>
NBL_PARTIAL_REQ_TOP(MICROFACET_CONF_ANISO)
struct SMicrofacetConfiguration<LS,Interaction,MicrofacetCache,Spectrum NBL_PARTIAL_REQ_BOT(MICROFACET_CONF_ANISO) > : SConfiguration<LS, Interaction, Spectrum>
#undef MICROFACET_CONF_ANISO
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = true;

    using base_type = SConfiguration<LS, Interaction, Spectrum>;

    using matrix3x3_type = matrix<typename base_type::scalar_type,3,3>;

    using isocache_type = typename MicrofacetCache::isocache_type;
    using anisocache_type = MicrofacetCache;
};

#define NBL_BXDF_CONFIG_ALIAS(TYPE,CONFIG) using TYPE = typename CONFIG::TYPE


#define BXDF_CONFIG_TYPE_ALIASES(Config) NBL_BXDF_CONFIG_ALIAS(scalar_type, Config);\
NBL_BXDF_CONFIG_ALIAS(vector2_type, Config);\
NBL_BXDF_CONFIG_ALIAS(vector3_type, Config);\
NBL_BXDF_CONFIG_ALIAS(monochrome_type, Config);\
NBL_BXDF_CONFIG_ALIAS(ray_dir_info_type, Config);\
NBL_BXDF_CONFIG_ALIAS(isotropic_interaction_type, Config);\
NBL_BXDF_CONFIG_ALIAS(anisotropic_interaction_type, Config);\
NBL_BXDF_CONFIG_ALIAS(sample_type, Config);\
NBL_BXDF_CONFIG_ALIAS(spectral_type, Config);\
NBL_BXDF_CONFIG_ALIAS(quotient_pdf_type, Config);\

#define MICROFACET_BXDF_CONFIG_TYPE_ALIASES(Config) BXDF_CONFIG_TYPE_ALIASES(Config);\
NBL_BXDF_CONFIG_ALIAS(matrix3x3_type, Config);\
NBL_BXDF_CONFIG_ALIAS(isocache_type, Config);\
NBL_BXDF_CONFIG_ALIAS(anisocache_type, Config);\

}
}
}

#endif
