#ifndef _NBL_BUILTIN_HLSL_FFT_COMMON_TWIDDLES_INCLUDED_
#define _NBL_BUILTIN_HLSL_FFT_COMMON_TWIDDLES_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/complex.hlsl"

#define LOW_TWIDDLE_BITS 6
#define MID_TWIDDLE_BITS 5
#define HIGH_TWIDDLE_BITS 4 

namespace nbl 
{
namespace hlsl
{
namespace fft::common {

    template<typename Scalar>
    complex_t<Scalar> getHigh(uint32_t highIdx); 

    template<typename Scalar>
    complex_t<Scalar> getMid(uint32_t midIdx); 

    template<typename Scalar>
    complex_t<Scalar> getLow(uint32_t lowIdx);

    // Define twiddles and getters for each Scalar template specialization
    #define float_t float32_t
    #define TYPED_NUMBER(N) NBL_CONCATENATE(N, f) // to add f after floating point numbers and avoid casting warnings and emitting ShaderFloat64 Caps
    #include <nbl/builtin/hlsl/fft/common/twiddles_impl.hlsl>
    #undef TYPED_NUMBER
    #undef float_t 

    #define float_t float64_t
    #define TYPED_NUMBER(N) N
    #include <nbl/builtin/hlsl/fft/common/twiddles_impl.hlsl>
    #undef TYPED_NUMBER
    #undef float_t
}
}
}


#endif