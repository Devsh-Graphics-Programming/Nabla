#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_FFT_ACCESSOR_CONCEPTS_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_FFT_ACCESSOR_CONCEPTS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/fft/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace fft
{
// The SharedMemoryAccessor MUST provide the following methods:
//      * void get(uint32_t index, inout uint32_t value);  
//      * void set(uint32_t index, in uint32_t value); 
//      * void workgroupExecutionAndMemoryBarrier();

#define NBL_CONCEPT_NAME FFTSharedMemoryAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (index_t, uint32_t)
#define NBL_CONCEPT_PARAM_2 (value_t, uint32_t)
NBL_CONCEPT_BEGIN(3)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define index_t NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define value_t NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.set(index_t, value_t)), is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.get(index_t, value_t)), is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.workgroupExecutionAndMemoryBarrier()), is_same_v, void))
);
#undef value_t
#undef index_t
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>



// The Accessor MUST provide a typename `Accessor::scalar_t`
// The Accessor MUST provide the following methods:
//     * void get(uint32_t index, inout complex_t<Scalar> value);
//     * void set(uint32_t index, in complex_t<Scalar> value);
//     * void memoryBarrier();

#define NBL_CONCEPT_NAME FFTAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (index_t, uint32_t)
#define NBL_CONCEPT_PARAM_2 (value_t, complex_t<typename T::scalar_t>)
NBL_CONCEPT_BEGIN(4)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define index_t NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define value_t NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_t))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.set(index_t, value_t)), is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.get(index_t, value_t)), is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.memoryBarrier()), is_same_v, void))
);
#undef value_t
#undef index_t
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
}

#endif