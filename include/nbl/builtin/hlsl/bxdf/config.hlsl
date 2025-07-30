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
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector2_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::vector3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::monochrome_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::matrix3x3_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::ray_dir_info_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisotropic_interaction_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::sample_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::spectral_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quotient_pdf_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::isocache_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::anisocache_type))
);
#undef conf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}

template<class LS, class Interaction, class Spectrum NBL_STRUCT_CONSTRAINABLE>
struct SConfiguration;

template<class LS, class Interaction, class Spectrum>
NBL_PARTIAL_REQ_TOP(LightSample<LS> && surface_interactions::Isotropic<Interaction> && !surface_interactions::Anisotropic<Interaction> && concepts::FloatingPointLikeVectorial<Spectrum>)
struct SConfiguration<LS,Interaction,Spectrum NBL_PARTIAL_REQ_BOT(LightSample<LS> && surface_interactions::Isotropic<Interaction> && !surface_interactions::Anisotropic<Interaction> && concepts::FloatingPointLikeVectorial<Spectrum>) >
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = false;

    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;

    using isotropic_interaction_type = Interaction;
    using anisotropic_interaction_type = surface_interactions::SAnisotropic<isotropic_interaction_type>;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
};

template<class LS, class Interaction, class Spectrum>
NBL_PARTIAL_REQ_TOP(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && concepts::FloatingPointLikeVectorial<Spectrum>)
struct SConfiguration<LS,Interaction,Spectrum NBL_PARTIAL_REQ_BOT(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && concepts::FloatingPointLikeVectorial<Spectrum>) >
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = true;

    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;

    using isotropic_interaction_type = typename Interaction::isotropic_interaction_type;
    using anisotropic_interaction_type = Interaction;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
};

template<class LS, class Interaction, class MicrofacetCache, class Spectrum NBL_STRUCT_CONSTRAINABLE>
struct SMicrofacetConfiguration;

template<class LS, class Interaction, class MicrofacetCache, class Spectrum>
NBL_PARTIAL_REQ_TOP(LightSample<LS> && surface_interactions::Isotropic<Interaction> && !surface_interactions::Anisotropic<Interaction> && CreatableIsotropicMicrofacetCache<MicrofacetCache> && !AnisotropicMicrofacetCache<MicrofacetCache> && concepts::FloatingPointLikeVectorial<Spectrum>)
struct SMicrofacetConfiguration<LS,Interaction,MicrofacetCache,Spectrum NBL_PARTIAL_REQ_BOT(LightSample<LS> && surface_interactions::Isotropic<Interaction> && !surface_interactions::Anisotropic<Interaction> && CreatableIsotropicMicrofacetCache<MicrofacetCache> && !AnisotropicMicrofacetCache<MicrofacetCache> && concepts::FloatingPointLikeVectorial<Spectrum>) >
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = false;

    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using monochrome_type = vector<scalar_type, 1>;

    using isotropic_interaction_type = Interaction;
    using anisotropic_interaction_type = surface_interactions::SAnisotropic<Interaction>;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = MicrofacetCache;
    using anisocache_type = SAnisotropicMicrofacetCache<MicrofacetCache>;
};

template<class LS, class Interaction, class MicrofacetCache, class Spectrum>
NBL_PARTIAL_REQ_TOP(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache> && concepts::FloatingPointLikeVectorial<Spectrum>)
struct SMicrofacetConfiguration<LS,Interaction,MicrofacetCache,Spectrum NBL_PARTIAL_REQ_BOT(LightSample<LS> && surface_interactions::Anisotropic<Interaction> && AnisotropicMicrofacetCache<MicrofacetCache> && concepts::FloatingPointLikeVectorial<Spectrum>) >
{
    NBL_CONSTEXPR_STATIC_INLINE bool IsAnisotropic = true;

    using scalar_type = typename LS::scalar_type;
    using ray_dir_info_type = typename LS::ray_dir_info_type;
    using vector2_type = vector<scalar_type, 2>;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type,3,3>;
    using monochrome_type = vector<scalar_type, 1>;

    using isotropic_interaction_type = typename Interaction::isotropic_interaction_type;
    using anisotropic_interaction_type = Interaction;
    using sample_type = LS;
    using spectral_type = Spectrum;
    using quotient_pdf_type = sampling::quotient_and_pdf<spectral_type, scalar_type>;
    using isocache_type = typename MicrofacetCache::isocache_type;
    using anisocache_type = MicrofacetCache;
};

}
}
}

#endif
