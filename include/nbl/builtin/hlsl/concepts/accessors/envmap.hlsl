#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_ENVMAP_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_ENVMAP_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace envmap
{
// declare concept
#define NBL_CONCEPT_NAME LuminanceReadAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (U)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (a,U)
#define NBL_CONCEPT_PARAM_1 (uv,uint32_t2)
#define NBL_CONCEPT_PARAM_2 (level,uint32_t)
#define NBL_CONCEPT_PARAM_3 (offset,uint32_t2)
// start concept
NBL_CONCEPT_BEGIN(4)
// need to be defined AFTER the concept begins
#define a NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define uv NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define level NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define offset NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((a.template get(uv,level,offset)) , ::nbl::hlsl::is_same_v, float32_t4>))
);
#undef offset
#undef level
#undef uv
#undef a
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template <typename T>
NBL_BOOL_CONCEPT WarpmapWriteAccessor = concepts::accessors::GenericWriteAccessor<T, float32_t2, uint32_t2>;

}
}
}
}

#endif