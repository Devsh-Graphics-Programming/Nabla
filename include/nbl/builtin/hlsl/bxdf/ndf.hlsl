// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

namespace dummy_impl
{
using sample_t = SLightSample<ray_dir_info::SBasic<float> >;
using interaction_t = surface_interactions::SAnisotropic<surface_interactions::SIsotropic<ray_dir_info::SBasic<float> > >;
using cache_t = SAnisotropicMicrofacetCache<SIsotropicMicrofacetCache<float> >;
struct DQuery   // nonsense struct, just put in all the functions to pass the ndf query concepts
{
    using scalar_type = float;

    scalar_type getNdf() NBL_CONST_MEMBER_FUNC { return 0; }
    scalar_type getLambdaL() NBL_CONST_MEMBER_FUNC { return 0; }
    scalar_type getLambdaV() NBL_CONST_MEMBER_FUNC { return 0; }

    scalar_type getG1over2NdotV() NBL_CONST_MEMBER_FUNC { return 0; }
    scalar_type getOrientedEta() NBL_CONST_MEMBER_FUNC { return 0; }
    scalar_type getDevshV() NBL_CONST_MEMBER_FUNC { return 0; }
    scalar_type getDevshL() NBL_CONST_MEMBER_FUNC { return 0; }
    BxDFClampMode getClampMode() NBL_CONST_MEMBER_FUNC { return BxDFClampMode::BCM_NONE; }
};
}

#define NBL_CONCEPT_NAME NDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (ndf, T)
#define NBL_CONCEPT_PARAM_1 (query, dummy_impl::DQuery)
#define NBL_CONCEPT_PARAM_2 (_sample, dummy_impl::sample_t)
#define NBL_CONCEPT_PARAM_3 (interaction, dummy_impl::interaction_t)
#define NBL_CONCEPT_PARAM_4 (cache, dummy_impl::cache_t)
NBL_CONCEPT_BEGIN(5)
#define ndf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define interaction NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template D<dummy_impl::cache_t>(cache)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template DG1<dummy_impl::DQuery>(query)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template DG1<dummy_impl::DQuery, dummy_impl::cache_t>(query, cache)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template correlated<dummy_impl::DQuery, dummy_impl::sample_t, dummy_impl::interaction_t>(query, _sample, interaction)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template G2_over_G1<dummy_impl::DQuery, dummy_impl::sample_t, dummy_impl::interaction_t, dummy_impl::cache_t>(query, _sample, interaction, cache)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef cache
#undef interaction
#undef _sample
#undef query
#undef ndf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
}

#endif