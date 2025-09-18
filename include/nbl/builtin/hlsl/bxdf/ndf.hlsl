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
}

#define NBL_CONCEPT_NAME NDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (ndf, T)
#define NBL_CONCEPT_PARAM_1 (quant_query, typename T::quant_query_type)
#define NBL_CONCEPT_PARAM_2 (_sample, dummy_impl::sample_t)
#define NBL_CONCEPT_PARAM_3 (interaction, dummy_impl::interaction_t)
#define NBL_CONCEPT_PARAM_4 (cache, dummy_impl::cache_t)
#define NBL_CONCEPT_PARAM_5 (dg1_query, typename T::dg1_query_type)
#define NBL_CONCEPT_PARAM_6 (g2_query, typename T::g2g1_query_type)
NBL_CONCEPT_BEGIN(7)
#define ndf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define quant_query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define interaction NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
#define dg1_query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_5
#define g2_query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_6
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_TYPE)(T::quant_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template D<dummy_impl::sample_t, dummy_impl::interaction_t, dummy_impl::cache_t>(quant_query, _sample, interaction, cache)), ::nbl::hlsl::is_same_v, typename T::quant_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template DG1<dummy_impl::sample_t, dummy_impl::interaction_t>(dg1_query, quant_query, _sample, interaction)), ::nbl::hlsl::is_same_v, typename T::quant_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template correlated<dummy_impl::sample_t, dummy_impl::interaction_t>(g2_query, quant_query, _sample, interaction)), ::nbl::hlsl::is_same_v, typename T::quant_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template G2_over_G1<dummy_impl::sample_t, dummy_impl::interaction_t, dummy_impl::cache_t>(g2_query, _sample, interaction, cache)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef g2_query
#undef dg1_query
#undef cache
#undef interaction
#undef _sample
#undef quant_query
#undef ndf
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
}

#endif