#ifndef _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_FFT_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"
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
//      * void get(uint32_t index, NBL_REF_ARG(uint32_t) value);  
//      * void set(uint32_t index, in uint32_t value); 
//      * void workgroupExecutionAndMemoryBarrier();

template<typename T, typename V=uint32_t, typename I=uint32_t>
NBL_BOOL_CONCEPT FFTSharedMemoryAccessor = concepts::accessors::GenericSharedMemoryAccessor<T,V,I>;

// The Accessor (for a small FFT) MUST provide the following methods:
//     * void get(uint32_t index, NBL_REF_ARG(complex_t<Scalar>) value);
//     * void set(uint32_t index, in complex_t<Scalar> value);

template<typename T, typename Scalar, typename I=uint32_t>
NBL_BOOL_CONCEPT FFTAccessor = concepts::accessors::GenericDataAccessor<T,complex_t<Scalar>,I>;

}
}
}
}

#endif