#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_FFT_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/fft/common.hlsl"

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
#define NBL_CONCEPT_PARAM_1 (index, uint32_t)
#define NBL_CONCEPT_PARAM_2 (val, uint32_t)
NBL_CONCEPT_BEGIN(3)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define index NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template set<uint32_t, uint32_t>(index, val)), is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template get<uint32_t, uint32_t>(index, val)), is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.workgroupExecutionAndMemoryBarrier()), is_same_v, void))
);
#undef val
#undef index
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>


// The Accessor (for a small FFT) MUST provide the following methods:
//     * void get(uint32_t index, inout complex_t<Scalar> value);
//     * void set(uint32_t index, in complex_t<Scalar> value);

#define NBL_CONCEPT_NAME FFTAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(Scalar)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (index, uint32_t)
#define NBL_CONCEPT_PARAM_2 (val, complex_t<Scalar>)
NBL_CONCEPT_BEGIN(3)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define index NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template set<complex_t<Scalar> >(index, val)), is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template get<complex_t<Scalar> >(index, val)), is_same_v, void))
);
#undef val
#undef index
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
}

#endif