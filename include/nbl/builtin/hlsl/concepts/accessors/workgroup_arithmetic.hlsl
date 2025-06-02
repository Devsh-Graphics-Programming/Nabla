#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_WORKGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_WORKGROUP_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup2
{

template<typename T, typename V, typename I=uint32_t>
NBL_BOOL_CONCEPT ArithmeticSharedMemoryAccessor = concepts::accessors::GenericSharedMemoryAccessor<T,V,I>;

template<typename T, typename V, typename I=uint32_t>
NBL_BOOL_CONCEPT ArithmeticReadOnlyDataAccessor = concepts::accessors::GenericReadAccessor<T,V,I>;

template<typename T, typename V, typename I=uint32_t>
NBL_BOOL_CONCEPT ArithmeticDataAccessor = concepts::accessors::GenericDataAccessor<T,V,I>;

}
}
}

#endif
