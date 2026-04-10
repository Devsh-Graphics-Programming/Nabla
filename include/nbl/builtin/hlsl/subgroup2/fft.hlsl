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
    static void FFT_loop(uint32_t stride, uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        const bool topHalf = bool(glsl::gl_SubgroupInvocationID() & stride);
        // Get twiddle with k = subgroupInvocation mod stride, halfN = stride
        const complex_t<Scalar> twiddle = fft2::twiddle<false, Scalar>(glsl::gl_SubgroupInvocationID() & (stride - 1), stride);
        [unroll]
        for (uint16_t channel = lowChannel; channel <= highChannel; channel++)
        {
            complex_t<Scalar> lo, hi;
            loAccessor.get(channel, lo);
            hiAccessor.get(channel, hi);
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
            fft2::DIF<Scalar>::radix2(twiddle, lo, hi);
            loAccessor.set(channel, lo);
            hiAccessor.set(channel, hi);
        }
    }

    template <typename InvocationElementsAccessor>
    static void __call(uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        // special first iteration
        const complex_t<Scalar> twiddle = fft2::twiddle<false, Scalar>(glsl::gl_SubgroupInvocationID(), SubgroupSize);
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
        [unroll]
        for (uint32_t stride = SubgroupSize >> 1; stride > 0; stride >>= 1)
            FFT_loop(stride, lowChannel, highChannel, loAccessor, hiAccessor);
    }

    // Interleaved versions of the above methods, required to implement the first steps in Interleaved DIF
    template <uint16_t NumSubgroupsLog2, typename InvocationElementsAccessor>
    static void FFT_loop(uint32_t elementStride, uint32_t threadStride, uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        const bool topHalf = bool(glsl::gl_SubgroupInvocationID() & threadStride);
        // Get twiddle with k = gl_SubgroupInvocationID() * NumSubgroups + gl_SubgroupID() mod elementStride, halfN = elementStride
        const uint32_t loLaneIndex = (glsl::gl_SubgroupInvocationID() << NumSubgroupsLog2) + glsl::gl_SubgroupID();
        const complex_t<Scalar> twiddle = fft2::twiddle<false, Scalar>(loLaneIndex & (elementStride - 1), elementStride);
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

    // Only uses subgroup methods, but is actually used at workgroup level
    template <uint16_t NumSubgroupsLog2, uint16_t WorkgroupSize, typename InvocationElementsAccessor>
    static void __callInterleaved(uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        // special first iteration
        // Get twiddle with k = gl_SubgroupInvocationID() * NumSubgroups + gl_SubgroupID() mod WorkgroupSize, halfN = WorkgroupSize
        const uint32_t loLaneIndex = (glsl::gl_SubgroupInvocationID() << NumSubgroupsLog2) + glsl::gl_SubgroupID();
        const complex_t<Scalar> twiddle = fft2::twiddle<false, Scalar>(loLaneIndex, WorkgroupSize); 
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
        [unroll]
        uint32_t threadStride = SubgroupSize >> 1;
        for (uint32_t elementStride = WorkgroupSize >> 1; elementStride > SubgroupSize; elementStride >>= 1)
        {
            FFT_loop<NumSubgroupsLog2>(elementStride, threadStride, lowChannel, highChannel, loAccessor, hiAccessor);
            threadStride >>= 1;
        }
    }
};


// ---------------------------------------- Radix 2 inverse transform - DIT -------------------------------------------------------

template<uint16_t SubgroupSize, typename Scalar, class device_capabilities>
struct FFT<SubgroupSize, true, Scalar, device_capabilities>
{
    template <typename InvocationElementsAccessor>
    static void FFT_loop(uint32_t stride, uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        const bool topHalf = bool(glsl::gl_SubgroupInvocationID() & stride);
        // Get twiddle with k = subgroupInvocation mod stride, halfN = stride
        const complex_t<Scalar> twiddle = fft2::twiddle<true, Scalar>(glsl::gl_SubgroupInvocationID() & (stride - 1), stride);

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

    template <typename InvocationElementsAccessor>
    static void __call(uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {                                                                        
        // Decimation in Time
        [unroll]
        for (uint32_t stride = 1; stride < SubgroupSize; stride <<= 1)
            FFT_loop(stride, lowChannel, highChannel, loAccessor, hiAccessor);
        
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

    // Interleaved versions of the above methods, required to implement the last steps in Interleaved DIT
    template <uint16_t NumSubgroupsLog2, typename InvocationElementsAccessor>
    static void FFT_loop(uint32_t elementStride, uint32_t threadStride, uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        const bool topHalf = bool(glsl::gl_SubgroupInvocationID() & threadStride);
        // Get twiddle with k = gl_SubgroupInvocationID() * NumSubgroups + gl_SubgroupID() mod elementStride, halfN = elementStride
        const uint32_t loLaneIndex = (glsl::gl_SubgroupInvocationID() << NumSubgroupsLog2) + glsl::gl_SubgroupID();
        const complex_t<Scalar> twiddle = fft2::twiddle<true, Scalar>(loLaneIndex & (elementStride - 1), elementStride);

        [unroll]
        for (uint16_t channel = lowChannel; channel <= highChannel; channel++)
        {
            complex_t<Scalar> lo, hi;
            loAccessor.get(channel, lo);
            hiAccessor.get(channel, hi);
            fft2::DIT<Scalar>::radix2(twiddle, lo, hi);

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
            loAccessor.set(channel, lo);
            hiAccessor.set(channel, hi);
        }
    }

    template <uint16_t NumSubgroupsLog2, uint16_t WorkgroupSize, typename InvocationElementsAccessor>
    static void __callInterleaved(uint16_t lowChannel, uint16_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor)
    {
        // Decimation in Time
        [unroll]
        uint32_t threadStride = SubgroupSize >> (NumSubgroupsLog2 - 1);
        for (uint32_t elementStride = SubgroupSize << 1; elementStride < WorkgroupSize; elementStride <<= 1)
        {
            FFT_loop<NumSubgroupsLog2>(elementStride, threadStride, lowChannel, highChannel, loAccessor, hiAccessor);
            threadStride <<= 1;
        }

        // special last iteration 
        // Get twiddle with k = gl_SubgroupInvocationID() * NumSubgroups + gl_SubgroupID() mod WorkgroupSize, halfN = WorkgroupSize
        const uint32_t loLaneIndex = (glsl::gl_SubgroupInvocationID() << NumSubgroupsLog2) + glsl::gl_SubgroupID();
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