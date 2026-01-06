#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_HIERARCHICAL_IMAGE_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_HIERARCHICAL_IMAGE_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{
namespace hierarchical_image
{
// declare concept
#define NBL_CONCEPT_NAME LuminanceReadAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (U)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (a,U)
#define NBL_CONCEPT_PARAM_1 (uv,uint32_t2)
#define NBL_CONCEPT_PARAM_2 (level,uint32_t)
// start concept
NBL_CONCEPT_BEGIN(3)
// need to be defined AFTER the concept begins
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define uv NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define level NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.template get(uv,level)) , ::nbl::hlsl::is_same_v, float32_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.template gather(uv,level)) , ::nbl::hlsl::is_same_v, float32_t4))
);
#undef level
#undef uv
#undef a
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

// declare concept
#define NBL_CONCEPT_NAME HierarchicalSampler
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (HierarchicalSamplerT)(ScalarT)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (sampler,HierarchicalSamplerT)
#define NBL_CONCEPT_PARAM_1 (coord,vector<ScalarT, 2>)
// start concept
NBL_CONCEPT_BEGIN(2)
// need to be defined AFTER the concept begins
#define sampler NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define coord NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((sampler.template sampleUvs(coord)) , ::nbl::hlsl::is_same_v, matrix<ScalarT, 4, 2>))
);
#undef sampler
#undef coord
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
}

#endif