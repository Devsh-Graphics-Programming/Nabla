#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_WARP_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_WARP_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"
#include "nbl/builtin/hlsl/fft/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{
  
template <typename C>
struct WarpResult
{
  C dst;
  float32_t density;
};

namespace concepts
{

// declare concept
#define NBL_CONCEPT_NAME WARP
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (U)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (warper,U)
#define NBL_CONCEPT_PARAM_1 (xi,typename U::domain_type)
#define NBL_CONCEPT_PARAM_2 (dst,typename U::codomain_type)
// start concept
NBL_CONCEPT_BEGIN(3)
#define warper NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define xi NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define dst NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(U::domain_type))
    ((NBL_CONCEPT_REQ_TYPE)(U::codomain_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((warper.template warp(xi)) , ::nbl::hlsl::is_same_v, WarpResult<typename U::codomain_type>))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((warper.template forwardDensity(xi)) , ::nbl::hlsl::is_same_v, float32_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((warper.template backwardDensity(dst)) , ::nbl::hlsl::is_same_v, float32_t))
);
#undef dst
#undef xi
#undef warper
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
}

#endif