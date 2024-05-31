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
    // Computes the kth element in the group of N roots of unity, where k = subgroupID mod N/2
    // Considering SubgroupSize and N as powers of 2, SubgroupSize <= 128, 2 <= N <= 2 * SubgroupSize
    // Making log2N a template parameter could help optimize this code further if necessary 

    // TODO: Add methods getLow/Mid/High that receive k and N, and are templated themselves on both subgroupSize and inverse
    // It's just as ugly but pushes the ugly code elsewhere. Maybe something else to scale to workgroup and global
    // TODO: N is always twice the stride. Could change parameters passed to simplify a bit
    template<typename Scalar, uint32_t log2SubgroupSize, bool inverse>
    complex_t<Scalar> getTwiddle(uint32_t log2N){
        // In this case all twiddles are just 1 so avoid lookups
        if (1 == log2N) {
            return {Scalar(1), Scalar(0)};                        
        }
        const uint32_t twiddleIdx = glsl::gl_SubgroupID();                
        // Can get away with only looking up lower 6 bits, N <= 64 = 2 ** LOW_TWIDDLE_BITS = 2 * SubgroupSize
        if (log2SubgroupSize < LOW_TWIDDLE_BITS){
            // arrayIdx is (twiddleIdx mod N/2) * 64 / N
            const uint32_t lowArrayIdx = (twiddleIdx & ((1u << (log2N - 1)) - 1)) * (1u << LOW_TWIDDLE_BITS - log2N);
            if (! inverse)
                return fft::common::getLow<Scalar>(lowArrayIdx);
            else 
                return conj(fft::common::getLow<Scalar>(lowArrayIdx));              
        }
        // If Subgroups are sized up to 128, N could be up to 256 so we also need to get the first 2 of the middle 5 bits
        else {
            uint32_t clampedLowerBits = max(LOW_TWIDDLE_BITS, log2N);
            uint32_t lowArrayIdx = (twiddleIdx & ((1u << (clampedLowerBits - 1)) - 1)) * (1u << LOW_TWIDDLE_BITS - clampedLowerBits);
            complex_t<Scalar> retVal = fft::common::getLow<Scalar>(lowArrayIdx); 
            if (log2N > LOW_TWIDDLE_BITS) {
                // Divide N by 2 ** LOW_TWIDDLE_BITS to get the next two significant bits, do the same for k
                const uint32_t log2N_p = log2N - LOW_TWIDDLE_BITS;
                const uint32_t twiddleIdx_p = twiddleIdx >> LOW_TWIDDLE_BITS;
                const uint32_t midArrayIdx = (twiddleIdx & ((1u << (log2N_p - 1)) - 1)) * (1u << MID_TWIDDLE_BITS - log2N_p);
                retVal *= fft::common::getMid<Scalar>(lowArrayIdx);               
            }
            if(! inverse)
                return retVal;
            else
                return conj(retVal);                       
        }                                                 
    }

    template<typename Scalar, uint32_t log2SubgroupSize, bool inverse>
    void FFT_loop(uint32_t log2Stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi) {
        const vector <Scalar, 4> loHiPacked = {lo.real(), lo.imag(), hi.real(), hi.imag()};
        vector <Scalar, 4> shuffledLoHiPacked = glsl::subgroupShuffleXor< vector <Scalar, 4> > (loHiPacked, 1u << log2Stride);
        lo = {shuffledLoHiPacked.x, shuffledLoHiPacked.y};
        hi = {shuffledLoHiPacked.z, shuffledLoHiPacked.w};
        fft::common::DIX<Scalar, inverse>::radix2Butterfly(getTwiddle<Scalar, log2SubgroupSize, inverse>(log2Stride + 1), lo, hi);   
    }


    template<typename Scalar, uint32_t log2SubgroupSize, bool inverse>
    void FFT(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi) {
        const uint32_t log2DoubleSubgroupSize = log2SubgroupSize + 1;
        // special first iteration
        if (! inverse)
            fft::common::DIX<Scalar, inverse>::radix2Butterfly(getTwiddle<Scalar, log2SubgroupSize, inverse>(log2DoubleSubgroupSize), lo, hi);                                                                                   
        
        // Decimation in Time
        if (inverse)
        for (uint32_t log2Stride = 0; log2Stride < log2SubgroupSize; log2Stride++)
            FFT_loop<Scalar, log2SubgroupSize, inverse>(log2Stride, lo, hi);
        // Decimation in Frequency
        else
        for (uint32_t log2Stride = log2SubgroupSize - 1; log2Stride > 0; log2Stride--)
            FFT_loop<Scalar, log2SubgroupSize, inverse>(log2Stride, lo, hi);
        
        // special last iteration 
        if (inverse){
            fft::common::DIX<Scalar, inverse>::radix2Butterfly(getTwiddle<Scalar, log2SubgroupSize, inverse>(log2DoubleSubgroupSize), lo, hi);
            lo <<= log2DoubleSubgroupSize;
            hi <<= log2DoubleSubgroupSize;
            
        }
    }   

}
}
}

#endif