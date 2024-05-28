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

    template<typename Scalar, uint32_t log2FFTSize, uint32_t WorkgroupSize, uint32_t SubgroupSize, bool inverse> 
    struct DIX {
        //Assumes I have precomputed twiddles, so just pass the index. 
        void radix2_butterfly(NBL_CONST_REF_ARG(complex_t<Scalar>) twiddle, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi){
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

    template<typename Scalar, uint32_t log2FFTSize, uint32_t WorkgroupSize, uint32_t SubgroupSize>
    using DIT = DIX<Scalar, log2FFTSize, WorkgroupSize, SubgroupSize, true>;

    template<typename Scalar, uint32_t log2FFTSize, uint32_t WorkgroupSize, uint32_t SubgroupSize>
    using DIF = DIX<Scalar, log2FFTSize, WorkgroupSize, SubgroupSize, false>;
}
}
}

#endif