#ifndef _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/fft/common/twiddles.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace fft::common {

    template<typename Scalar, bool inverse> 
    struct DIX {
        //Assumes I have precomputed twiddles, so just pass the index. 
        static void radix2Butterfly(NBL_CONST_REF_ARG(complex_t<Scalar>) twiddle, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi){
            //Decimation in time - inverse           
            if (inverse) {
                complex_t<Scalar> wHi = twiddle * hi;
                hi = lo - wHi;
                lo += wHi;            
            }
            //Decimation in frequency - forward   
            else {
                complex_t<Scalar> diff = lo - hi;
                lo += hi;
                hi = twiddle * diff; 
            }
        };                                              
    }

    /* 
    template<typename Scalar>
    using DIT = DIX<Scalar, true>;

    template<typename Scalar>
    using DIF = DIX<Scalar, false>;
    */
}
}
}

#endif