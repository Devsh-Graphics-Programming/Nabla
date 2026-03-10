#ifndef _NBL_BUILTIN_HLSL_HIERARCHICAL_IMAGE_ACCESSORS_INCLUDED_
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
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (U)(ScalarT)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (a,U)
#define NBL_CONCEPT_PARAM_1 (coord,uint32_t2)
#define NBL_CONCEPT_PARAM_2 (level,uint32_t)
// start concept
NBL_CONCEPT_BEGIN(3)
// need to be defined AFTER the concept begins
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define coord NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define level NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.template load(coord,level)) , ::nbl::hlsl::is_same_v, ScalarT))
);
#undef level
#undef coord
#undef a
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

// gatherUvs return 4 UVs in a square for manual bilinear interpolation with differentiability
// declare concept
#define NBL_CONCEPT_NAME WarpAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (WarpAccessorT)(ScalarT)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (accessor,WarpAccessorT)
#define NBL_CONCEPT_PARAM_1 (coord,vector<uint32_t, 2>)
#define NBL_CONCEPT_PARAM_2 (val, matrix<ScalarT, 4, 2>)
#define NBL_CONCEPT_PARAM_3 (interpolant, vector<ScalarT, 2>)
// start concept
NBL_CONCEPT_BEGIN(4)
// need to be defined AFTER the concept begins
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define coord NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define interpolant NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.gatherUv(coord, val, interpolant)), ::nbl::hlsl::is_same_v, void))
);
#undef accessor
#undef coord
#undef val
#undef interpolant
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
}

#endif
