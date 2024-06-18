#ifndef _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_FFT_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace fft 
{

// Computes the kth element in the group of N roots of unity
// Notice 0 <= k < N/2, rotating counterclockwise in the forward (DIF) transform and clockwise in the inverse (DIT)
template<bool inverse, typename Scalar>
complex_t<Scalar> twiddle(uint32_t k, uint32_t N)
{
    complex_t<Scalar> retVal;
    const Scalar kthRootAngleRadians = 2.f * numbers::pi<Scalar> * Scalar(k) / Scalar(N);
    retVal.real( cos(kthRootAngleRadians) );
    if (! inverse)
        retVal.imag( sin(kthRootAngleRadians) );
    else
        retVal.imag( sin(-kthRootAngleRadians) );
    return retVal;                         
}

template<bool inverse, typename Scalar> 
struct DIX 
{ 
    static void radix2(NBL_CONST_REF_ARG(complex_t<Scalar>) twiddle, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi)
    {
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

template<typename Scalar>
using DIT = DIX<true, Scalar>;

template<typename Scalar>
using DIF = DIX<false, Scalar>;
}
}
}

#endif