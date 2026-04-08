#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/fft2/common.hlsl>

#ifndef _NBL_BUILTIN_HLSL_WORKGROUP2_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP2_FFT_INCLUDED_

// ------------------------------- COMMON -----------------------------------------

namespace nbl
{
namespace hlsl
{
namespace workgroup2
{
namespace fft
{
// Minimum size (in number of uint32_t elements) of the workgroup shared memory array needed for the FFT
template<typename Scalar, bool Interleaved>
uint32_t minimumSharedMemoryDWORDs(uint16_t workgroupSizeLog2)
{
    NBL_IF_CONSTEXPR(Interleaved)
    {
        return (sizeof(complex_t<Scalar>) / sizeof(uint32_t)) << (workgroupSizeLog2 + 1);
    }
    else
    {
        return (sizeof(complex_t<Scalar>) / sizeof(uint32_t)) << workgroupSizeLog2;
    }  
}

template<uint16_t _ElementsPerInvocation, uint16_t _SubgroupSizeLog2, uint16_t _WorkgroupSizeLog2, uint16_t _ShuffledElementsPerRound, typename _Scalar NBL_PRIMARY_REQUIRES(_ElementsPerInvocation > 1 && !(_ElementsPerInvocation & 1) && _WorkgroupSizeLog2 >= 5)
struct ConstevalParameters
{
    using scalar_t = _Scalar;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocation = _ElementsPerInvocation;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = _SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t FFTTotalSize = ElementsPerInvocation * (uint32_t(1) << WorkgroupSizeLog2);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ShuffledElementsPerRound = _ShuffledElementsPerRound;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedMemoryDWORDs = ShuffledElementsPerRound * (sizeof(complex_t<scalar_t>) / sizeof(uint32_t)) << WorkgroupSizeLog2;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(1) << WorkgroupSizeLog2;
};
}

struct OptimalFFTParameters
{
    uint16_t elementsPerInvocation : 8;
    uint16_t workgroupSizeLog2 : 8;

    // Used to check if the parameters returned by `optimalFFTParameters` are valid
    bool areValid()
    {
        return elementsPerInvocation > 0 && workgroupSizeLog2 > 0;
    }
};

/**
* @brief Returns the best parameters (according to our metric) to run an FFT
*
* @param [in] maxWorkgroupSize The max number of threads that can be launched in a single workgroup
* @param [in] inputArrayLength The length of the array to run an FFT on
* @param [in] subgroupSize Number of threads running in a subgroup
*/
inline OptimalFFTParameters optimalFFTParameters(uint32_t maxWorkgroupSize, uint32_t inputArrayLength, uint32_t subgroupSize)
{
    NBL_CONSTEXPR_STATIC OptimalFFTParameters invalidParameters = { 0 , 0 };

    if (subgroupSize < 2 || maxWorkgroupSize < subgroupSize || inputArrayLength <= subgroupSize)
        return invalidParameters;
    // Pad inputarrayLength to size that FFT algo handles
    const uint32_t FFTLength = hlsl::fft2::padDimension(inputArrayLength, subgroupSize);
    // Round maxWorkgroupSize down to PoT
    const uint32_t actualMaxWorkgroupSize = hlsl::roundDownToPoT(maxWorkgroupSize);
    // Max number of threads that can run the FFT
    uint32_t maxThreads = FFTLength / 2;
    // Factors of 3 and 5 do not contribute to the amount of max threads since those are handled in-register to keep everything PoT later
    maxThreads /= maxThreads % 3 ? 1 : 3;
    maxThreads /= maxThreads % 5 ? 1 : 5;
    // Both are PoT
    const uint16_t workgroupSizeLog2 = findMSB(min(maxThreads, actualMaxWorkgroupSize));

    // Parameters are valid if the workgroup size is at most half of the FFT Length and at least as big as the subgroupSize
    if ((FFTLength >> workgroupSizeLog2) <= 1 || subgroupSize > (1u << workgroupSizeLog2))
    {
        return invalidParameters;
    }

    const uint16_t elementsPerInvocation = FFTLength >> workgroupSizeLog2;
    const OptimalFFTParameters retVal = { elementsPerInvocation, workgroupSizeLog2 };

    return retVal;
}

namespace impl
{
template<uint16_t Radix2ElementsPerInvocationLog2, uint16_t WorkgroupSizeLog2, uint16_t ExtraPrimeFactor>
struct FFTIndexingUtilsHelper
{
    // Maps the lane of index `laneIdx` at the end of the DIF diagram to its corresponding frequency position as an element of the DFT.
    static uint32_t mapLaneToFreq(uint32_t laneIdx)
    {
        NBL_IF_CONSTEXPR(ExtraPrimeFactor > 1)
        {
            const uint32_t radix2mask = (1 << Radix2FFTSizeLog2) - 1;
            return ExtraPrimeFactor * hlsl::bitReverseAs<uint32_t>(laneIdx, Radix2FFTSizeLog2) + (laneIdx >> Radix2FFTSizeLog2);
        }
    else
    {
        return hlsl::bitReverseAs<uint32_t>(laneIdx, Radix2FFTSizeLog2);
        }
    }

    // Implements fast division by 3 or 5, needed by `mapFreqtoLane`
    static uint32_t fastDiv(uint32_t x)
    {
        NBL_IF_CONSTEXPR(ExtraPrimeFactor == 3)
        {
            return (x * 43691u) >> 17; // valid for x <= 98303
        }
    else // ExtraPrimeFactor == 5
    {
        return (x * 52429u) >> 18; // valid for x <= 81919
        }
    }

    // Inverse of `mapLaneToFreq`. Maps a frequency index `freqIdx` into the DFT to the lane in the DIF diagram that outputs it.
    static uint32_t mapFreqToLane(uint32_t freqIdx)
    {
        NBL_IF_CONSTEXPR(ExtraPrimeFactor > 1)
        {
            const uint32_t divByPrimeFactor = fastDiv(freqIdx);
            return hlsl::bitReverseAs<uint32_t>(divByPrimeFactor, Radix2FFTSizeLog2) + ((freqIdx - ExtraPrimeFactor * divByPrimeFactor) << Radix2FFTSizeLog2);
        }
    else
    {
        return hlsl::bitReverseAs<uint32_t>(freqIdx, Radix2FFTSizeLog2);
        }
    }

    // log2(ElementsPerInvocation / ExtraPrimeFactor)
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Radix2FFTSizeLog2 = WorkgroupSizeLog2 + Radix2ElementsPerInvocationLog2;
    // Size of the full FFT if no mixed radix used, otherwise size of the sub-FFTs computed after the first radix-3/5 step in the DIF forward FFT
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Radix2FFTSize = uint32_t(1) << (Radix2FFTSizeLog2);
    // Total size of the FFT computed
    NBL_CONSTEXPR_STATIC_INLINE uint32_t FFTSize = ExtraPrimeFactor * Radix2FFTSize;
};
}

template<uint16_t Radix2ElementsPerInvocationLog2, uint16_t WorkgroupSizeLog2, uint16_t ExtraPrimeFactor = 1>
struct FFTIndexingUtils
{
    using helper_t = impl::FFTIndexingUtilsHelper<Radix2ElementsPerInvocationLog2, WorkgroupSizeLog2, ExtraPrimeFactor>;

    // Maps the array index 'arrayIdx' of the output of an FFT in workgroup-linear order (meaning all threads write their local element 0 contiguously
    // and in ascending order by threadIndex, then element 1 and so on) to its corresponding frequency position as an element of the DFT.
    static uint32_t mapArrayToFreq(uint32_t arrayIdx)
    {
        return helper_t::mapLaneToFreq(circularBitShiftLeftLower<WorkgroupSizeLog2 + 1>(arrayIdx));
    }

    // Maps a frequency index 'freqIdx' into the DFT to its corresponding position in the output array of an FFT when written in workgroup-linear order.
    static uint32_t mapFreqToArray(uint32_t freqIdx)
    {
        return circularBitShiftRightLower<WorkgroupSizeLog2 + 1>(helper_t::mapFreqToLane(freqIdx));
    }

    // Mirrors an index about the Nyquist frequency in the DFT order
    static uint32_t getDFTMirrorIndex(uint32_t freqIdx)
    {
        return (FFTSize - freqIdx) & (FFTSize - 1);
    }

    // Given an index `arrayIdx` of an element into the output array of an FFT, get the index into the same array of the element corresponding 
    // to its negative frequency
    static uint32_t getNablaMirrorIndex(uint32_t arrayIdx)
    {
        return mapFreqToArray(getDFTMirrorIndex(mapArrayToFreq(arrayIdx)));
    }

    // log2(ElementsPerInvocation / ExtraPrimeFactor)
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Radix2FFTSizeLog2 = helper_t::Radix2FFTSizeLog2;
    // Size of the full FFT if no mixed radix used, otherwise size of the sub-FFTs computed after the first radix-3/5 step in the DIF forward FFT
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Radix2FFTSize = helper_t::Radix2FFTSize;
    // Total size of the FFT computed
    NBL_CONSTEXPR_STATIC_INLINE uint32_t FFTSize = helper_t::FFTSize;
};

}
}
}
// ------------------------------- END COMMON ---------------------------------------------




#endif
