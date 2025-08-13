// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"

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
struct DQuery {};
struct DLightSample {};
struct DInteraction {};
struct DMicrofacetCache {};
}

#define NBL_CONCEPT_NAME NDF
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (ndf, T)
#define NBL_CONCEPT_PARAM_1 (query, dummy_impl::DQuery)
#define NBL_CONCEPT_PARAM_2 (_sample, dummy_impl::DLightSample)
#define NBL_CONCEPT_PARAM_3 (interaction, dummy_impl::DInteraction)
#define NBL_CONCEPT_PARAM_4 (cache, dummy_impl::DMicrofacetCache)
NBL_CONCEPT_BEGIN(5)
#define ndf NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define _sample NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define interaction NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
#define cache NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_4
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    // ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template D<dummy_impl::DMicrofacetCache>(cache)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    // ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template DG1<dummy_impl::DQuery>(query)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    // ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template DG1<dummy_impl::DQuery, dummy_impl::DMicrofacetCache>(query, cache)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    // ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template correlated<dummy_impl::DQuery, dummy_impl::DLightSample, dummy_impl::DInteraction>(query, _sample, interaction)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    // ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((ndf.template G2_over_G1<dummy_impl::DQuery, dummy_impl::DLightSample, dummy_impl::DInteraction, dummy_impl::DMicrofacetCache>(query, _sample, interaction, cache)), ::nbl::hlsl::is_same_v, typename T::scalar_type))
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