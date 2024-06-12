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
    // TODO: specialize subgorupShuffleXor to different Scalar types. As of right now this would only support 1-wide scalars (no vectors)
    template<typename Scalar, bool inverse>
    void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi) 
    {
        const vector <Scalar, 4> loHiPacked = {lo.real(), lo.imag(), hi.real(), hi.imag()};
        vector <Scalar, 4> shuffledLoHiPacked = glsl::subgroupShuffleXor< vector <Scalar, 4> > (loHiPacked, stride);
        lo.real(shuffledLoHiPacked.x);
        lo.imag(shuffledLoHiPacked.y);
        hi.real(shuffledLoHiPacked.z);
        hi.imag(shuffledLoHiPacked.w);
        // Get twiddle with k = subgroupID mod stride, N = 2 * stride
        fft::DIX<Scalar, inverse>::radix2(fft::twiddle<Scalar, inverse>(glsl::gl_SubgroupID() & (stride - 1), stride << 1), lo, hi);   
    }

    // When doing a 1D FFT of size N, we call N/2 threads to do the work, where each thread of index 0 <= k < N/2 is in charge of computing two elements
    // in the output array, those of index k and k + N/2. The first is "lo" and the latter is "hi"
    template<typename Scalar, uint32_t SubgroupSize, bool inverse>
    void FFT(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi) 
    {
        const uint32_t doubleSubgroupSize = SubgroupSize << 1;
        // special first iteration
        if (! inverse)
            fft::DIF<Scalar>::radix2(fft::twiddle<Scalar, false>(glsl::gl_SubgroupID(), doubleSubgroupSize), lo, hi);                                                                               
        
        // Decimation in Time
        if (inverse)
        for (uint32_t stride = 1; stride < SubgroupSize; stride <<= 1)
            FFT_loop<Scalar, true>(stride, lo, hi);
        // Decimation in Frequency
        else
        for (uint32_t stride = SubgroupSize >> 1; stride > 0; stride >>= 1)
            FFT_loop<Scalar, false>(stride, lo, hi);
        
        // special last iteration 
        if (inverse){
            fft::DIT<Scalar>::radix2(fft::twiddle<Scalar, true>(glsl::gl_SubgroupID(), doubleSubgroupSize), lo, hi);
            divides_assign< complex_t<Scalar> > divAss;
            divAss(lo, doubleSubgroupSize);
            divAss(hi, doubleSubgroupSize);            
        } 
    }   

}
}
}

#endif