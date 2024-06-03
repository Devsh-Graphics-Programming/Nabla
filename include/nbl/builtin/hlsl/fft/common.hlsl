#ifndef _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/math/complex.hlsl"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/math/numbers.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace fft {

    // Computes the kth element in the group of N roots of unity
    template<typename Scalar, bool inverse>
    complex_t<Scalar> twiddle(uint32_t k, uint32_t N){
        complex_t<Scalar> retVal;
        retVal.real( cos(2.f * math::Numbers<Scalar>::pi * float(k) / N) );
        if (! inverse)
            retVal.imag( sin(2.f * math::Numbers<Scalar>::pi * float(k) / N) );
        else
            retVal.imag( sin(-2.f * math::Numbers<Scalar>::pi * float(k) / N) );
        return retVal;                         
    }

    template<typename Scalar, bool inverse> 
    struct DIX { 
        static void radix2(NBL_CONST_REF_ARG(complex_t<Scalar>) twiddle, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi){
            plus_assign< complex_t<Scalar> > plusAss;
            //Decimation in time - inverse           
            if (inverse) {
                complex_t<Scalar> wHi = twiddle * hi;
                hi = lo - wHi;
                plusAss(lo, wHi);            
            }
            //Decimation in frequency - forward   
            else {
                complex_t<Scalar> diff = lo - hi;
                plusAss(lo, hi);
                hi = twiddle * diff; 
            }
        }                                              
    };

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