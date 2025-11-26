#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_GENERIC_SHARED_DATA_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_GENERIC_SHARED_DATA_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"

namespace nbl
{
namespace hlsl
{
namespace concepts
{
namespace accessors
{

#define NBL_CONCEPT_NAME GenericSharedMemoryAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(V)(I)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (val, V)
#define NBL_CONCEPT_PARAM_2 (index, I)

#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define index NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template set<V,I>(index, val)), is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template get<V,I>(index, val)), is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.workgroupExecutionAndMemoryBarrier()), is_same_v, void))
);
#undef val
#undef index
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME GenericReadAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(V)(I)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (val, V)
#define NBL_CONCEPT_PARAM_2 (index, I)
NBL_CONCEPT_BEGIN(3)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define index NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template get<V,I>(index, val)), is_same_v, void))
);
#undef val
#undef index
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

#define NBL_CONCEPT_NAME GenericWriteAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(V)(I)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (val, V)
#define NBL_CONCEPT_PARAM_2 (index, I)
NBL_CONCEPT_BEGIN(3)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define index NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template set<V,I>(index, val)), is_same_v, void))
);
#undef val
#undef index
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T, typename V, typename I=uint32_t>
NBL_BOOL_CONCEPT GenericDataAccessor = GenericWriteAccessor<T,V,I> && GenericWriteAccessor<T,V,I>;

}
}
}
}

#endif
