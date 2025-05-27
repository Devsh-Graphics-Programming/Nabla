#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_WORKGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_WORKGROUP_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup2
{

template<typename T, typename V, typename I>
NBL_BOOL_CONCEPT ArithmeticSharedMemoryAccessor = concepts::accessors::GenericSharedMemoryAccessor<T,V,I>;

#define NBL_CONCEPT_NAME ArithmeticReadOnlyDataAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(V)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (index, uint32_t)
#define NBL_CONCEPT_PARAM_2 (val, V)
NBL_CONCEPT_BEGIN(3)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define index NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template get<V>(index, val)), is_same_v, void))
);
#undef val
#undef index
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

template<typename T, typename V, typename I=uint32_t>
NBL_BOOL_CONCEPT ArithmeticDataAccessor = concepts::accessors::GenericDataAccessor<T,V,I>;

}
}
}

#endif
