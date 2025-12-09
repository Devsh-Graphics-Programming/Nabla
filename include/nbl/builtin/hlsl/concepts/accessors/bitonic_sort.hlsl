#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_BITONIC_SORT_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_BITONIC_SORT_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace bitonic_sort
{
// The SharedMemoryAccessor MUST provide the following methods:
//  * void get(uint32_t index, NBL_REF_ARG(uint32_t) value);
//  * void set(uint32_t index, in uint32_t value);
//  * void workgroupExecutionAndMemoryBarrier();
template<typename T, typename V = uint32_t, typename I = uint32_t>
NBL_BOOL_CONCEPT BitonicSortSharedMemoryAccessor = concepts::accessors::GenericSharedMemoryAccessor<T, V, I>;

// The Accessor MUST provide the following methods:
//  * void get(uint32_t index, NBL_REF_ARG(pair<KeyType, ValueType>) value);
//  * void set(uint32_t index, in pair<KeyType, ValueType> value);
template<typename T, typename KeyType, typename ValueType, typename I = uint32_t>
NBL_BOOL_CONCEPT BitonicSortAccessor = concepts::accessors::GenericDataAccessor<T, pair<KeyType, ValueType>, I>;

}
}
}
}
#endif
