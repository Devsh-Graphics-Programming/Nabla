#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_WARP_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_WARP_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"
#include "nbl/builtin/hlsl/fft/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace concepts
{

// declare concept
#define NBL_CONCEPT_NAME WARP
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (U)(C)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (warp,U)
#define NBL_CONCEPT_PARAM_1 (uv,float32_t2)
#define NBL_CONCEPT_PARAM_2 (out,C)
// start concept
NBL_CONCEPT_BEGIN(3)
#define warp NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define uv NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define out NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.template warp(uv)) , ::nbl::hlsl::is_same_v, C))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.template forwardDensity(uv)) , ::nbl::hlsl::is_same_v, float32_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.template backwardDensity(out)) , ::nbl::hlsl::is_same_v, float32_t))
);
#undef out
#undef warp
#undef uv
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}

#endif