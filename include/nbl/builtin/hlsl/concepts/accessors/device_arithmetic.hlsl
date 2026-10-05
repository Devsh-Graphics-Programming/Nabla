#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_DEVICE_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_DEVICE_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{

template<typename T, typename V, typename I=uint32_t>
NBL_BOOL_CONCEPT ArithmeticSharedMemoryAccessor = concepts::accessors::GenericSharedMemoryAccessor<T,V,I>;

template<typename T, typename V, typename I=uint32_t>
NBL_BOOL_CONCEPT ArithmeticReadOnlyDataAccessor = concepts::accessors::GenericReadAccessor<T,V,I>;

template<typename T, typename V, typename I=uint32_t>
NBL_BOOL_CONCEPT ArithmeticDataAccessor = concepts::accessors::GenericDataAccessor<T,V,I>;

#define NBL_CONCEPT_NAME DeviceReductionsAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(V)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (val, V)
#define NBL_CONCEPT_PARAM_2 (index, uint64_t)
NBL_CONCEPT_BEGIN(3)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define index NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(concepts::accessors::GenericDataAccessor, T, V, uint64_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.atomicMax(index, val)), is_same_v, V))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.atomicExchange(index, val)), is_same_v, V))
);
#undef val
#undef index
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

// TODO: as counter, maybe just increment 1 always?
#define NBL_CONCEPT_NAME WorkgroupCounterAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (val, uint32_t)
#define NBL_CONCEPT_PARAM_2 (index, uint64_t)
NBL_CONCEPT_BEGIN(3)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define index NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.atomicAdd(index, val)), is_same_v, uint32_t))
);
#undef val
#undef index
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}

#endif
