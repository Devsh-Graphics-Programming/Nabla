#ifndef _NBL_BUILTIN_HLSL_SUBGROUP2_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP2_FFT_INCLUDED_

#include "nbl/builtin/hlsl/fft2/common.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_shuffle.hlsl"
#include "nbl/builtin/hlsl/concepts/accessors/fft.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup2
{

// -----------------------------------------------------------------------------------------------------------------------------------------------------------------
template<uint16_t SubgroupSize, bool Inverse, typename Scalar, class device_capabilities=void>
struct FFT
{
    template <typename InvocationElementsAccessor>
    static void __call(uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor);
};

// ---------------------------------------- Radix 2 forward transform - DIF -------------------------------------------------------

template<uint16_t SubgroupSize, typename Scalar, class device_capabilities>
struct FFT<SubgroupSize, false, Scalar, device_capabilities>
{
    template <typename InvocationElementsAccessor>
    static void FFT_loop(uint32_t threadStride, uint16_t lowChannel, uint16_t highChannel, NBL_CONST_REF_ARG(complex_t<Scalar>) twiddle, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        const bool topHalf = bool(glsl::gl_SubgroupInvocationID() & threadStride);
        [unroll]
        for (uint16_t channel = lowChannel; channel <= highChannel; channel++)
        {
            complex_t<Scalar> lo, hi;
            loAccessor.get(channel, lo);
            hiAccessor.get(channel, hi);
            const vector <Scalar, 2> toTrade = topHalf ? vector <Scalar, 2>(lo.real(), lo.imag()) : vector <Scalar, 2>(hi.real(), hi.imag());
            const vector <Scalar, 2> exchanged = glsl::subgroupShuffleXor< vector <Scalar, 2> >(toTrade, threadStride);
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
            fft2::DIF<Scalar>::radix2(twiddle, lo, hi);
            loAccessor.set(channel, lo);
            hiAccessor.set(channel, hi);
        }
    }

    template <bool ShareTwiddles, typename InvocationElementsAccessor>
    static void __call(uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        // special first iteration
        complex_t<Scalar> twiddle = fft2::twiddle<false, Scalar>(glsl::gl_SubgroupInvocationID(), SubgroupSize);
        [unroll]
        for (uint16_t channel = lowChannel; channel <= highChannel; channel++)
        {
            complex_t<Scalar> lo, hi;
            loAccessor.get(channel, lo);
            hiAccessor.get(channel, hi);
            fft2::DIF<Scalar>::radix2(twiddle, lo, hi);
            loAccessor.set(channel, lo);
            hiAccessor.set(channel, hi);
        }                                                                        
        
        // Decimation in Frequency
        // Compute all twiddles at the start, then reshare them among threads
        if (ShareTwiddles)
        {
            uint32_t iteration = 1;
            [unroll]
            for (uint32_t threadStride = SubgroupSize >> 1; threadStride > 0; threadStride >>= 1)
            {
                const vector <Scalar, 2> toTrade = vector <Scalar, 2>(twiddle.real(), twiddle.imag());
                const vector <Scalar, 2> otherTwiddle = glsl::subgroupShuffle< vector <Scalar, 2> >(toTrade, (glsl::gl_SubgroupInvocationID() & (threadStride - 1)) << iteration);
                twiddle.real(otherTwiddle.x);
                twiddle.imag(otherTwiddle.y);
                FFT_loop(threadStride, lowChannel, highChannel, twiddle, loAccessor, hiAccessor);
                iteration++;
            }
        }
        // Recompute twiddles at every step
        else 
        {
            [unroll]
            for (uint32_t threadStride = SubgroupSize >> 1; threadStride > 0; threadStride >>= 1)
            {
                // Get twiddle with k = subgroupInvocation mod threadStride, halfN = threadStride
                const complex_t<Scalar> twiddle = fft2::twiddle<false, Scalar>(glsl::gl_SubgroupInvocationID() & (threadStride - 1), threadStride);
                FFT_loop(threadStride, lowChannel, highChannel, twiddle, loAccessor, hiAccessor);
            }
        } 
    }

    // Only uses subgroup methods, but is actually used at workgroup level. Used by the interleaved workgroup FFT at bigger than subgroup strides
    template <uint16_t NumSubgroupsLog2, uint16_t WorkgroupSize, typename InvocationElementsAccessor>
    static void __callInterleaved(uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        const uint32_t loLaneIndex = (glsl::gl_SubgroupInvocationID() << NumSubgroupsLog2) + glsl::gl_SubgroupID();
        // special first iteration
        [unroll]
        for (uint16_t channel = lowChannel; channel <= highChannel; channel++)
        {
            // Get twiddle with k = gl_SubgroupInvocationID() * NumSubgroups + gl_SubgroupID() mod WorkgroupSize, halfN = WorkgroupSize
            const complex_t<Scalar> twiddle = fft2::twiddle<false, Scalar>(loLaneIndex, WorkgroupSize);
            complex_t<Scalar> lo, hi;
            loAccessor.get(channel, lo);
            hiAccessor.get(channel, hi);
            fft2::DIF<Scalar>::radix2(twiddle, lo, hi);
            loAccessor.set(channel, lo);
            hiAccessor.set(channel, hi);
        }

        // Decimation in Frequency
        uint32_t threadStride = SubgroupSize >> 1;
        [unroll]
        for (uint32_t elementStride = WorkgroupSize >> 1; elementStride > SubgroupSize; elementStride >>= 1)
        {
            // Get twiddle with k = gl_SubgroupInvocationID() * NumSubgroups + gl_SubgroupID() mod elementStride, halfN = elementStride
            const complex_t<Scalar> twiddle = fft2::twiddle<false, Scalar>(loLaneIndex & (elementStride - 1), elementStride);
            FFT_loop(threadStride, lowChannel, highChannel, twiddle, loAccessor, hiAccessor);
            threadStride >>= 1;
        }
    }
};


// ---------------------------------------- Radix 2 inverse transform - DIT -------------------------------------------------------

template<uint16_t SubgroupSize, typename Scalar, class device_capabilities>
struct FFT<SubgroupSize, true, Scalar, device_capabilities>
{
    template <typename InvocationElementsAccessor>
    static void FFT_loop(uint32_t stride, uint16_t lowChannel, uint16_t highChannel, NBL_CONST_REF_ARG(complex_t<Scalar>) twiddle, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        const bool topHalf = bool(glsl::gl_SubgroupInvocationID() & stride);

        [unroll]
        for (uint16_t channel = lowChannel; channel <= highChannel; channel++)
        {
            complex_t<Scalar> lo, hi;
            loAccessor.get(channel, lo);
            hiAccessor.get(channel, hi);
            fft2::DIT<Scalar>::radix2(twiddle, lo, hi);

            const vector <Scalar, 2> toTrade = topHalf ? vector <Scalar, 2>(lo.real(), lo.imag()) : vector <Scalar, 2>(hi.real(), hi.imag());
            const vector <Scalar, 2> exchanged = glsl::subgroupShuffleXor< vector <Scalar, 2> >(toTrade, stride);
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
            loAccessor.set(channel, lo);
            hiAccessor.set(channel, hi);
        }
    }

    template <bool ShareTwiddles, typename InvocationElementsAccessor>
    static void __call(uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {                                                                        
        // Decimation in Time
        // Compute all twiddles at the start, then shuffle them
        if (ShareTwiddles)
        {
            const complex_t<Scalar> ownedTwiddle = fft2::twiddle<true, Scalar>(glsl::gl_SubgroupInvocationID(), SubgroupSize);
            uint32_t reverseStride = SubgroupSize;
            [unroll]
            for (uint32_t threadStride = 1; threadStride < SubgroupSize; threadStride <<= 1)
            {
                const vector <Scalar, 2> toTrade = vector <Scalar, 2>(ownedTwiddle.real(), ownedTwiddle.imag());
                const vector <Scalar, 2> otherTwiddle = glsl::subgroupShuffle< vector <Scalar, 2> >(toTrade, (glsl::gl_SubgroupInvocationID() & (threadStride - 1)) * reverseStride);
                const complex_t<Scalar> twiddle = { otherTwiddle.x , otherTwiddle.y };
                FFT_loop(threadStride, lowChannel, highChannel, twiddle, loAccessor, hiAccessor);
                reverseStride >>= 1;
            }
        }
        // Compute each twiddle at each iteration
        else
        {
            [unroll]
            for (uint32_t threadStride = 1; threadStride < SubgroupSize; threadStride <<= 1)
            {
                // Get twiddle with k = subgroupInvocation mod threadStride, halfN = threadStride
                const complex_t<Scalar> twiddle = fft2::twiddle<true, Scalar>(glsl::gl_SubgroupInvocationID() & (threadStride - 1), threadStride);
                FFT_loop(threadStride, lowChannel, highChannel, twiddle, loAccessor, hiAccessor);
            } 
        }
        
        // special last iteration 
        const complex_t<Scalar> twiddle = fft2::twiddle<true, Scalar>(glsl::gl_SubgroupInvocationID(), SubgroupSize);
        [unroll]
        for (uint16_t channel = lowChannel; channel <= highChannel; channel++)
        {
            complex_t<Scalar> lo, hi;
            loAccessor.get(channel, lo);
            hiAccessor.get(channel, hi);
            fft2::DIT<Scalar>::radix2(twiddle, lo, hi);
            loAccessor.set(channel, lo);
            hiAccessor.set(channel, hi);
        }
    }

    template <uint16_t NumSubgroupsLog2, uint16_t WorkgroupSize, typename InvocationElementsAccessor>
    static void __callInterleaved(uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        const uint32_t loLaneIndex = (glsl::gl_SubgroupInvocationID() << NumSubgroupsLog2) + glsl::gl_SubgroupID();
        // Decimation in Time
        uint32_t threadStride = SubgroupSize >> (NumSubgroupsLog2 - 1);
        [unroll]
        for (uint32_t elementStride = SubgroupSize << 1; elementStride < WorkgroupSize; elementStride <<= 1)
        {
            // Get twiddle with k = gl_SubgroupInvocationID() * NumSubgroups + gl_SubgroupID() mod elementStride, halfN = elementStride
            const complex_t<Scalar> twiddle = fft2::twiddle<true, Scalar>(loLaneIndex & (elementStride - 1), elementStride);
            FFT_loop<NumSubgroupsLog2>(threadStride, lowChannel, highChannel, loAccessor, hiAccessor);
            threadStride <<= 1;
        }

        // special last iteration 
        // Get twiddle with k = gl_SubgroupInvocationID() * NumSubgroups + gl_SubgroupID() mod WorkgroupSize, halfN = WorkgroupSize
        const complex_t<Scalar> twiddle = fft2::twiddle<true, Scalar>(loLaneIndex, WorkgroupSize);
        [unroll]
        for (uint16_t channel = lowChannel; channel <= highChannel; channel++)
        {
            complex_t<Scalar> lo, hi;
            loAccessor.get(channel, lo);
            hiAccessor.get(channel, hi);
            fft2::DIT<Scalar>::radix2(twiddle, lo, hi);
            loAccessor.set(channel, lo);
            hiAccessor.set(channel, hi);
        }
    }
};


} //namespace subgroup2
} //namespace hlsl
} //namespace nbl

#endif