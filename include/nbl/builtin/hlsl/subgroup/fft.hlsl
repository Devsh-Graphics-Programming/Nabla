#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_FFT_INCLUDED_

#include "nbl/builtin/hlsl/fft/common.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------
template<uint16_t K, bool Inverse, typename Scalar, class device_capabilities=void>
struct FFT
{
    static void __call(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi);
};

// ---------------------------------------- Radix 2 forward transform - DIF -------------------------------------------------------

template<typename Scalar, class device_capabilities>
struct FFT<2, false, Scalar, device_capabilities>
{
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi)
    {
        const bool topHalf = (glsl::gl_SubgroupInvocationID() & stride) != 0;
        const vector <Scalar, 2> toTrade = topHalf ? vector <Scalar, 2>(lo.real(), lo.imag()) : vector <Scalar, 2>(hi.real(), hi.imag());
        const vector <Scalar, 2> exchanged = glsl::subgroupShuffleXor< vector <Scalar, 2> > (toTrade, stride);
        if (topHalf)
        {
            lo.real(exchanged.x);
            lo.imag(exchanged.y);
        }
        else
        {
            hi.real(exchanged.x);
            hi.imag(exchanged.y);
        }
        // Get twiddle with k = subgroupInvocation mod stride, N = 2 * stride
        fft::DIF<Scalar>::radix2(fft::twiddle<false, Scalar>(glsl::gl_SubgroupInvocationID() & (stride - 1), stride << 1), lo, hi); 
    }

    static void __call(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi) 
    {
        const uint32_t subgroupSize = glsl::gl_SubgroupSize();  //This is N/2
        const uint32_t doubleSubgroupSize = subgroupSize << 1;  //This is N
    
        // special first iteration
        fft::DIF<Scalar>::radix2(fft::twiddle<false, Scalar>(glsl::gl_SubgroupInvocationID(), doubleSubgroupSize), lo, hi);                                                                                   
        
        // Decimation in Frequency
        for (uint32_t stride = subgroupSize >> 1; stride > 0; stride >>= 1)
            FFT_loop(stride, lo, hi);
    }
};


// ---------------------------------------- Radix 2 inverse transform - DIT -------------------------------------------------------

template<typename Scalar, class device_capabilities>
struct FFT<2, true, Scalar, device_capabilities>
{
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi)
    {
        // Get twiddle with k = subgroupInvocation mod stride, N = 2 * stride
        fft::DIT<Scalar>::radix2(fft::twiddle<true, Scalar>(glsl::gl_SubgroupInvocationID() & (stride - 1), stride << 1), lo, hi);   

        const bool topHalf = (glsl::gl_SubgroupInvocationID() & stride) != 0;
        const vector <Scalar, 2> toTrade = topHalf ? vector <Scalar, 2>(lo.real(), lo.imag()) : vector <Scalar, 2>(hi.real(), hi.imag());
        const vector <Scalar, 2> exchanged = glsl::subgroupShuffleXor< vector <Scalar, 2> > (toTrade, stride);
        if (topHalf)
        {
            lo.real(exchanged.x);
            lo.imag(exchanged.y);
        }
        else
        {
            hi.real(exchanged.x);
            hi.imag(exchanged.y);
        }
    }

    static void __call(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi) 
    {
        const uint32_t subgroupSize = glsl::gl_SubgroupSize();  //This is N/2
        const uint32_t doubleSubgroupSize = subgroupSize << 1;  //This is N                                                                           
        
        // Decimation in Time
        for (uint32_t stride = 1; stride < subgroupSize; stride <<= 1)
            FFT_loop(stride, lo, hi);
        
        // special last iteration 
        fft::DIT<Scalar>::radix2(fft::twiddle<true, Scalar>(glsl::gl_SubgroupInvocationID(), doubleSubgroupSize), lo, hi);
        divides_assign< complex_t<Scalar> > divAss;
        divAss(lo, doubleSubgroupSize);
        divAss(hi, doubleSubgroupSize);
    }
};


}
}
}

#endif