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

// The DFT (and DFT) have two different formulations. One of them doesn't divide in DFT and divides by N in the IDFT. This makes the determinant of the DFT
// sqrt(N) and the determinant of the IDFT as sqrt(N)^(-1). This formulation is problematic, for example, when performing FFT Convolution of images when 
// using half-precision, since if N is big this can make the FFT along the second axis (and sometimes even the first!) exceed the representable range and become
// NaNs. The other formulation divides by sqrt(N) on both the DFT and IDFT, giving a determinant of 1 for both transforms. This can be used to avoid
// overflow.
// These policies describe different ways of avoiding overflow by dividing at different moments through the algorithm. 
struct DivisionPolicy
{
    // No division performed at any step of the FFT
    NBL_CONSTEXPR_STATIC_INLINE uint16_t NoDivision = 0;
    // Divides the array by sqrt(FFTSize) at the time of the last workgroup barrier before subgroupFFT (forward) or at the time of the first workgroup barrier
    // after subgroupFFT (inverse).
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DivBySqrtHalfway = NoDivision + 1;
    // Divides the array by sqrt(FFTSize) right at the end of the algorithm
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DivBySqrtAtEnd = DivBySqrtHalfway + 1;
    // Divides the array by sqrt(FFTSize) by considering `sqrt(FFTSize) = a * b`, dividing by `a` halfway (as described in `DivBySqrtHalfway`) and then dividing 
    // by `b` at the end. `a` and `b` are chosen so that their weight is proportional to the number of butterflies before the division.
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DivBySqrtByParts = DivBySqrtAtEnd + 1;
    // The three following all perform divisions in the same manner as their counterparts above, but they divide the array by `FFTSize`.
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DivByFullSizeHalfway = DivBySqrtByParts + 1;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DivByFullSizeAtEnd = DivByFullSizeHalfway + 1;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DivByFullSizeByParts = DivByFullSizeAtEnd + 1;
};

// TODO: Separate parallelFFTs from Channels as two different concepts (dictates FFTSize)
template<uint16_t _ElementsPerInvocation, uint16_t _SubgroupSizeLog2, uint16_t _WorkgroupSizeLog2, uint16_t _ShuffledChannelsPerRound, bool _Interleaved, uint16_t _DivisionPolicy, typename _Scalar> //NBL_PRIMARY_REQUIRES(_ElementsPerInvocation > 1 && !(_ElementsPerInvocation & 1) && _WorkgroupSizeLog2 >= 5)
struct ConstevalParameters
{
    using scalar_t = _Scalar;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocation = _ElementsPerInvocation;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Channels = ElementsPerInvocation >> 1;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSizeLog2 = _SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t SubgroupSize = 1 << SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(1) << WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t NumSubgroupsLog2 = WorkgroupSizeLog2 - SubgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t FFTTotalSize = ElementsPerInvocation * (uint32_t(1) << WorkgroupSizeLog2);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ShuffledChannelsPerRound = _ShuffledChannelsPerRound;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t ShuffleRounds = mpl::ceil_div_v<Channels, ShuffledChannelsPerRound>;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedMemoryDWORDs = ShuffledChannelsPerRound * ((sizeof(complex_t<scalar_t>) / sizeof(uint32_t)) << (WorkgroupSizeLog2 + (_Interleaved ? 1 : 0)));

    NBL_CONSTEXPR_STATIC_INLINE uint16_t DivisionPolicy = _DivisionPolicy;
};
} //namespace fft

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
} // namespace impl

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

// TODO: Implement when doing 2D FFTConv
template<uint16_t ElementsPerInvocationLog2, uint16_t WorkgroupSizeLog2>
struct FFTMirrorTradeUtils;

}
}
}
// ------------------------------- END COMMON ---------------------------------------------

// ------------------------------- HLSL ONLY ---------------------------------------------

#ifdef __HLSL_VERSION

#include "nbl/builtin/hlsl/subgroup2/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"

namespace nbl
{
namespace hlsl
{
namespace workgroup2
{

//-------------- ---------------------------------------- UTILS --------------------------------------------------------

namespace fft
{
namespace impl
{

template <uint32_t WorkgroupSize, typename SharedMemoryAdaptor, typename Scalar, typename InvocationElementsAccessor>
struct exchangeValues
{
    static void __call(uint32_t threadID, uint32_t ownedSmemIndex, uint32_t lowChannel, uint32_t highChannel, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor, uint32_t stride, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor, NBL_REF_ARG(bool) pingPong)
    {
        const bool topHalf = bool(threadID & stride);
        const uint32_t writeIndex = pingPong ? ownedSmemIndex ^ stride : ownedSmemIndex;
        const uint32_t readIndex = pingPong ? ownedSmemIndex : ownedSmemIndex ^ stride;
        // Write elements to sharedmem
        uint32_t adaptorOffset = 0;
        [unroll]
        for (uint32_t channel = lowChannel; channel <= highChannel; channel++)
        {
            complex_t<Scalar> lo, hi;
            loAccessor.get(channel, lo);
            hiAccessor.get(channel, hi);
            vector<Scalar, 2> toExchange = topHalf ? vector<Scalar, 2>(lo.real(), lo.imag()) : vector<Scalar, 2>(hi.real(), hi.imag());
            sharedmemAdaptor.template set<vector<Scalar, 2> >(adaptorOffset | writeIndex, toExchange);

            adaptorOffset += WorkgroupSize;
        }
        // Wait until all writes are done before reading
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();

        // Read elements from sharedmem
        adaptorOffset = 0;
        [unroll]
        for (uint32_t channel = lowChannel; channel <= highChannel; channel++)
        {
            vector<Scalar, 2> exchanged;
            sharedmemAdaptor.template get<vector<Scalar, 2> >(adaptorOffset | readIndex, exchanged);
            complex_t<Scalar> complex_exchanged = { exchanged.x, exchanged.y };
            if (topHalf)
            {
                loAccessor.set(channel, complex_exchanged);
            }
            else
            {
                hiAccessor.set(channel, complex_exchanged);
            }

            adaptorOffset += WorkgroupSize;
        }
    }

    static void __callInterleaved()
    {
        // TODO
    }
};

} //namespace impl
} //namespace fft

//-------------- ------------------------------------ END UTILS --------------------------------------------------------

template<bool Inverse, typename consteval_params_t, class device_capabilities = void>
struct FFT;

// Non-interleaved (shuffle after every butterfly) forward FFT
template<uint16_t ElementsPerInvocation, uint16_t SubgroupSizeLog2, uint16_t WorkgroupSizeLog2, uint16_t ShuffledChannelsPerRound, uint16_t DivisionPolicy, typename Scalar, class device_capabilities>
struct FFT<false, fft::ConstevalParameters<ElementsPerInvocation, SubgroupSizeLog2, WorkgroupSizeLog2, ShuffledChannelsPerRound, false, DivisionPolicy, Scalar>, device_capabilities>
{
    using consteval_parameters_t = fft::ConstevalParameters<ElementsPerInvocation, SubgroupSizeLog2, WorkgroupSizeLog2, ShuffledChannelsPerRound, false, DivisionPolicy, Scalar>;
    using scalar_t = typename consteval_parameters_t::scalar_t;

    template<typename InvocationElementsAccessor, typename SharedMemoryAdaptor>
    static void FFT_loop(uint32_t stride, uint32_t threadID, NBL_REF_ARG(uint32_t) ownedSmemIndex, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        const uint32_t ShuffleRounds = consteval_parameters_t::ShuffleRounds;
        const uint16_t Channels = consteval_parameters_t::Channels;
        // Get twiddle with k = threadID mod stride, halfN = stride
        const complex_t<scalar_t> twiddle = hlsl::fft::twiddle<false, scalar_t>(threadID & (stride - 1), stride);

        bool pingPong = false;
        // Unrolling this loop increases register pressure. Why? Who knows. 
        // It's not like it can't reuse the registers, and calls to exchangeValues are inlined anyway.
        //[unroll]
        for (uint32_t round = 0; round < ShuffleRounds; round++)
        {
            if (round)
                pingPong = !pingPong; // ping pong on sharedmem to avoid barriering - this eploits that we XOR with the same stride every consecutive round
            const uint32_t lowChannel = round * ShuffledChannelsPerRound;
            const uint32_t highChannel = min(Channels, lowChannel + ShuffledChannelsPerRound) - 1;
            [unroll]
            for (uint32_t channel = lowChannel; channel <= highChannel; channel++)
            {
                complex_t<scalar_t> lo, hi;
                loAccessor.get(channel, lo);
                hiAccessor.get(channel, hi);
                fft2::DIF<scalar_t>::radix2(twiddle, lo, hi);
                loAccessor.set(channel, lo);
                hiAccessor.set(channel, hi);
            }

            fft::impl::exchangeValues<consteval_parameters_t::WorkgroupSize, SharedMemoryAdaptor, scalar_t, InvocationElementsAccessor>::__call(threadID, ownedSmemIndex, lowChannel, highChannel, loAccessor, hiAccessor, stride >> 1, sharedmemAdaptor, pingPong);
        }
        // After the last exchangeValues, the memory we just read from is now owned by us, so update
        ownedSmemIndex = pingPong ? ownedSmemIndex : ownedSmemIndex ^ (stride >> 1);
    }

    template<typename InvocationElementsAccessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        const uint16_t Channels = consteval_parameters_t::Channels;
        const uint16_t SubgroupSize = consteval_parameters_t::SubgroupSize;
        const uint16_t WorkgroupSize = consteval_parameters_t::WorkgroupSize;

        // Get workgroup threadID
        const uint32_t threadID = uint32_t(workgroup::SubgroupContiguousIndex());

        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps
        if (WorkgroupSize > SubgroupSize)
        {
            // Set up the memory adaptor
            using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, ShuffledChannelsPerRound * WorkgroupSize>;
            adaptor_t sharedmemAdaptor;
            sharedmemAdaptor.accessor = sharedmemAccessor;

            uint32_t ownedSmemIndex = threadID;
            // NOT unrolling this loop increases register pressure????
            [unroll]
            for (uint32_t stride = WorkgroupSize; stride > SubgroupSize; stride >>= 1)
            {
                FFT_loop(stride, threadID, ownedSmemIndex, loAccessor, hiAccessor, sharedmemAdaptor);
            }

            // Remember to update the accessor's state
            sharedmemAccessor = sharedmemAdaptor.accessor;
        }
        // Subgroup-sized FFT
        subgroup2::FFT<SubgroupSize, false, Scalar, device_capabilities>::__call(0, Channels - 1, loAccessor, hiAccessor);
    }
};

// Non-interleaved (shuffle after every butterfly) inverse FFT
template<uint16_t ElementsPerInvocation, uint16_t SubgroupSizeLog2, uint16_t WorkgroupSizeLog2, uint16_t ShuffledChannelsPerRound, uint16_t DivisionPolicy, typename Scalar, class device_capabilities>
struct FFT<true, fft::ConstevalParameters<ElementsPerInvocation, SubgroupSizeLog2, WorkgroupSizeLog2, ShuffledChannelsPerRound, false, DivisionPolicy, Scalar>, device_capabilities>
{
    using consteval_parameters_t = fft::ConstevalParameters<ElementsPerInvocation, SubgroupSizeLog2, WorkgroupSizeLog2, ShuffledChannelsPerRound, false, DivisionPolicy, Scalar>;
    using scalar_t = typename consteval_parameters_t::scalar_t;

    template<typename InvocationElementsAccessor, typename SharedMemoryAdaptor>
    static void FFT_loop(uint32_t stride, uint32_t threadID, NBL_REF_ARG(uint32_t) ownedSmemIndex, NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        const uint32_t ShuffleRounds = consteval_parameters_t::ShuffleRounds;
        const uint16_t Channels = consteval_parameters_t::Channels;
        // Get twiddle with k = threadID mod stride, halfN = stride
        const complex_t<scalar_t> twiddle = hlsl::fft::twiddle<true, scalar_t>(threadID & ((stride << 1) - 1), stride << 1);

        bool pingPong = false;
        //[unroll]
        for (uint32_t round = 0; round < ShuffleRounds; round++)
        {
            if (round)
                pingPong = !pingPong; // ping pong on sharedmem to avoid barriering - this eploits that we XOR with the same stride every consecutive round
            const uint32_t lowChannel = round * ShuffledChannelsPerRound;
            const uint32_t highChannel = min(Channels, lowChannel + ShuffledChannelsPerRound) - 1;

            fft::impl::exchangeValues<consteval_parameters_t::WorkgroupSize, SharedMemoryAdaptor, scalar_t, InvocationElementsAccessor>::__call(threadID, ownedSmemIndex, lowChannel, highChannel, loAccessor, hiAccessor, stride, sharedmemAdaptor, pingPong);
            
            [unroll]
            for (uint32_t channel = lowChannel; channel <= highChannel; channel++)
            {
                complex_t<scalar_t> lo, hi;
                loAccessor.get(channel, lo);
                hiAccessor.get(channel, hi);
                fft2::DIT<scalar_t>::radix2(twiddle, lo, hi);
                loAccessor.set(channel, lo);
                hiAccessor.set(channel, hi);
            }
        }
        // After the last exchangeValues, the memory we just read from is now owned by us, so update
        ownedSmemIndex = pingPong ? ownedSmemIndex : ownedSmemIndex ^ (stride >> 1);
    }

    template<typename InvocationElementsAccessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(InvocationElementsAccessor) loAccessor, NBL_REF_ARG(InvocationElementsAccessor) hiAccessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        const uint16_t Channels = consteval_parameters_t::Channels;
        const uint16_t SubgroupSize = consteval_parameters_t::SubgroupSize;
        const uint16_t WorkgroupSize = consteval_parameters_t::WorkgroupSize;

        // Subgroup-sized FFT at the start
        subgroup2::FFT<SubgroupSize, false, Scalar, device_capabilities>::__call(0, Channels - 1, loAccessor, hiAccessor);

        // Get workgroup threadID
        const uint32_t threadID = uint32_t(workgroup::SubgroupContiguousIndex());

        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps
        if (WorkgroupSize > SubgroupSize)
        {
            // Set up the memory adaptor
            using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, ShuffledChannelsPerRound * WorkgroupSize>;
            adaptor_t sharedmemAdaptor;
            sharedmemAdaptor.accessor = sharedmemAccessor;

            uint32_t ownedSmemIndex = threadID;
            [unroll]
            for (uint32_t stride = SubgroupSize; stride < WorkgroupSize; stride <<= 1)
            {
                FFT_loop(stride, threadID, ownedSmemIndex, loAccessor, hiAccessor, sharedmemAdaptor);
            }

            // Remember to update the accessor's state
            sharedmemAccessor = sharedmemAdaptor.accessor;
        }
    }
};

} //namespace workgroup2
} //namespace hlsl
} //namespace nbl

#endif


#endif
