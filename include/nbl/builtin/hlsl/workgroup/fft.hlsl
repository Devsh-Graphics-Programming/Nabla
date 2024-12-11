#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/fft/common.hlsl>

#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_

// ------------------------------- COMMON -----------------------------------------

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace fft
{

template<uint16_t _ElementsPerInvocationLog2, uint16_t _WorkgroupSizeLog2, typename _Scalar NBL_PRIMARY_REQUIRES(_ElementsPerInvocationLog2 > 0 && _WorkgroupSizeLog2 >= 5)
struct ConstevalParameters
{
    using scalar_t = _Scalar;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocationLog2 = _ElementsPerInvocationLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = _WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t TotalSize = uint32_t(1) << (ElementsPerInvocationLog2 + WorkgroupSizeLog2);

    NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocation = uint16_t(1) << ElementsPerInvocationLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = uint16_t(1) << WorkgroupSizeLog2;

    // Required size (in number of uint32_t elements) of the workgroup shared memory array needed for the FFT
    NBL_CONSTEXPR_STATIC_INLINE uint32_t SharedMemoryDWORDs = (sizeof(complex_t<scalar_t>) / sizeof(uint32_t)) << WorkgroupSizeLog2;
};

}
}
}
} 
// ------------------------------- END COMMON ---------------------------------------------

// -------------------------------- CPP ONLY ----------------------------------------------

#ifndef __HLSL_VERSION

namespace nbl
{
namespace hlsl
{
namespace workgroup
{
namespace fft
{

struct OptimalFFTParameters
{
    uint16_t elementsPerInvocationLog2;
    uint16_t workgroupSizeLog2;
};

inline OptimalFFTParameters optimalFFTParameters(const uint32_t maxWorkgroupSize, uint32_t inputArrayLength)
{
    // This is the logic found in core::roundUpToPoT to get the log2
    const uint16_t workgroupSizeLog2 = 1u + findMSB(min(inputArrayLength / 2, maxWorkgroupSize) - 1u);
    const uint16_t elementsPerInvocationLog2 = 1u + findMSB(max((inputArrayLength >> workgroupSizeLog2) - 1u, 1u));
    const OptimalFFTParameters retVal = { elementsPerInvocationLog2, workgroupSizeLog2 };
    return retVal;
}

}
}
}
}
// ------------------------------- END CPP ONLY -------------------------------------------

// ------------------------------- HLSL ONLY ----------------------------------------------

#else 

#include "nbl/builtin/hlsl/subgroup/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/shuffle.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/concepts/accessors/fft.hlsl"

// Caveats
// - Sin and Cos in HLSL take 32-bit floats. Using this library with 64-bit floats works perfectly fine, but DXC will emit warnings
//   This also means that you don't really get the full precision of 64-bit since twiddle factors are only 32-bit precision

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{
namespace fft
{

// ---------------------------------- Utils -----------------------------------------------

// No need to expose these
namespace impl
{
    template<typename SharedMemoryAdaptor, typename Scalar>
    struct exchangeValues
    {
        static void __call(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, uint32_t stride, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
        {
            const bool topHalf = bool(threadID & stride);
            // Pack into float vector because ternary operator does not support structs
            vector<Scalar, 2> exchanged = topHalf ? vector<Scalar, 2>(lo.real(), lo.imag()) : vector<Scalar, 2>(hi.real(), hi.imag());
            shuffleXor<SharedMemoryAdaptor, vector<Scalar, 2> >(exchanged, stride, sharedmemAdaptor);
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
    };

    template<uint16_t N, uint16_t H>
    enable_if_t<(H <= N) && (N < 32), uint32_t> circularBitShiftRightHigher(uint32_t i)
    {
        // Highest H bits are numbered N-1 through N - H
        // N - H is then the middle bit
        // Lowest bits numbered from 0 through N - H - 1
        NBL_CONSTEXPR_STATIC_INLINE uint32_t lowMask = (1 << (N - H)) - 1;
        NBL_CONSTEXPR_STATIC_INLINE uint32_t midMask = 1 << (N - H);
        NBL_CONSTEXPR_STATIC_INLINE uint32_t highMask = ~(lowMask | midMask);

        uint32_t low = i & lowMask;
        uint32_t mid = i & midMask;
        uint32_t high = i & highMask;

        high >>= 1;
        mid <<= H - 1;

        return mid | high | low;
    }

    template<uint16_t N, uint16_t H>
    enable_if_t<(H <= N) && (N < 32), uint32_t> circularBitShiftLeftHigher(uint32_t i)
    {
        // Highest H bits are numbered N-1 through N - H
        // N - 1 is then the highest bit, and N - 2 through N - H are the middle bits
        // Lowest bits numbered from 0 through N - H - 1
        NBL_CONSTEXPR_STATIC_INLINE uint32_t lowMask = (1 << (N - H)) - 1;
        NBL_CONSTEXPR_STATIC_INLINE uint32_t highMask = 1 << (N - 1);
        NBL_CONSTEXPR_STATIC_INLINE uint32_t midMask = ~(lowMask | highMask);

        uint32_t low = i & lowMask;
        uint32_t mid = i & midMask;
        uint32_t high = i & highMask;

        mid <<= 1;
        high >>= H - 1;

        return mid | high | low;
    }
} //namespace impl

template<uint16_t ElementsPerInvocationLog2, uint16_t WorkgroupSizeLog2>
struct FFTIndexingUtils
{
    // This function maps the index `idx` in the output array of a Nabla FFT to the index `freqIdx` in the DFT such that `DFT[freqIdx] = NablaFFT[idx]`
    // This is because Cooley-Tukey + subgroup operations end up spewing out the outputs in a weird order
    static uint32_t getDFTIndex(uint32_t outputIdx)
    {
        return impl::circularBitShiftRightHigher<FFTSizeLog2, FFTSizeLog2 - ElementsPerInvocationLog2 + 1>(glsl::bitfieldReverse<uint32_t>(outputIdx) >> (32 - FFTSizeLog2));
    }

    // This function maps the index `freqIdx` in the DFT to the index `idx` in the output array of a Nabla FFT such that `DFT[freqIdx] = NablaFFT[idx]`
    // It is essentially the inverse of `getDFTIndex`
    static uint32_t getNablaIndex(uint32_t freqIdx)
    {
        return glsl::bitfieldReverse<uint32_t>(impl::circularBitShiftLeftHigher<FFTSizeLog2, FFTSizeLog2 - ElementsPerInvocationLog2 + 1>(freqIdx)) >> (32 - FFTSizeLog2);
    }

    // Mirrors an index about the Nyquist frequency in the DFT order
    static uint32_t getDFTMirrorIndex(uint32_t idx)
    {
        return (FFTSize - idx) & (FFTSize - 1);
    }

    // Given an index `idx` of an element into the Nabla FFT, get the index into the Nabla FFT of the element corresponding to its negative frequency
    static uint32_t getNablaMirrorIndex(uint32_t idx)
    {
        return getNablaIndex(getDFTMirrorIndex(getDFTIndex(idx)));
    }

    // When unpacking an FFT of two packed signals, given a `localElementIndex` representing a `globalElementIndex` you need its "mirror index" to unpack the value at 
    // NablaFFT[globalElementIndex].
    // The function above has you covered in that sense, but what also happens is that not only does the thread holding `NablaFFT[globalElementIndex]` need its mirror value
    // but also the thread holding said mirror value will at the same time be trying to unpack `NFFT[someOtherIndex]` and need the mirror value of that. 
    // As long as this unpacking is happening concurrently and in order (meaning the local element index - the higher bits - of `globalElementIndex` and `someOtherIndex` is the
    // same) then this function returns both the SubgroupContiguousIndex of the other thread AND the local element index of *the mirror* of `someOtherIndex` 
    struct NablaMirrorLocalInfo
    {
        uint32_t otherThreadID;
        uint32_t mirrorLocalIndex;
    };
    
    static NablaMirrorLocalInfo getNablaMirrorLocalInfo(uint32_t globalElementIndex)
    {
        const uint32_t otherElementIndex = FFTIndexingUtils::getNablaMirrorIndex(globalElementIndex);
        const uint32_t mirrorLocalIndex = otherElementIndex / WorkgroupSize;
        const uint32_t otherThreadID = otherElementIndex & (WorkgroupSize - 1);
        const NablaMirrorLocalInfo info = { otherThreadID, mirrorLocalIndex };
        return info;
    }

    // Like the above, but return global indices instead.
    struct NablaMirrorGlobalInfo
    {
        uint32_t otherThreadID;
        uint32_t mirrorGlobalIndex;
    };

    static NablaMirrorGlobalInfo getNablaMirrorGlobalInfo(uint32_t globalElementIndex)
    {
        const uint32_t otherElementIndex = FFTIndexingUtils::getNablaMirrorIndex(globalElementIndex);
        const uint32_t mirrorGlobalIndex = glsl::bitfieldInsert<uint32_t>(otherElementIndex, workgroup::SubgroupContiguousIndex(), 0, uint32_t(WorkgroupSizeLog2));
        const uint32_t otherThreadID = otherElementIndex & (WorkgroupSize - 1);
        const NablaMirrorGlobalInfo info = { otherThreadID, mirrorGlobalIndex };
        return info;
    }

    NBL_CONSTEXPR_STATIC_INLINE uint16_t FFTSizeLog2 = ElementsPerInvocationLog2 + WorkgroupSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t FFTSize = uint32_t(1) << FFTSizeLog2;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t WorkgroupSize = uint32_t(1) << WorkgroupSizeLog2;
};

template<uint16_t ElementsPerInvocationLog2, uint16_t WorkgroupSizeLog2>
struct FFTMirrorTradeUtils
{
    using indexing_utils_t = FFTIndexingUtils<ElementsPerInvocationLog2, WorkgroupSizeLog2>;
    using mirror_info_t = typename indexing_utils_t::NablaMirrorGlobalInfo;
    // If trading elements when, for example, unpacking real FFTs, you might do so from within your accessor or from outside. 
    // If doing so from within your accessor, particularly if using a preloaded accessor, you might want to do this yourself by
    // using FFTIndexingUtils::getNablaMirrorTradeInfo and trading the elements yourself (an example of how to set this up is given in
    // the FFT Bloom example, in the `fft_mirror_common.hlsl` file).
    // If you're doing this from outside your preloaded accessor then you might want to use this method instead.
    // Note: you can still pass a preloaded accessor as `arrayAccessor` here, it's just that you're going to be doing extra computations for the indices.
    template<typename scalar_t, typename fft_array_accessor_t, typename shared_memory_adaptor_t>
    static complex_t<scalar_t> getNablaMirror(uint32_t globalElementIndex, fft_array_accessor_t arrayAccessor, shared_memory_adaptor_t sharedmemAdaptor)
    {
        const mirror_info_t mirrorInfo = indexing_utils_t::getNablaMirrorGlobalInfo(globalElementIndex);
        complex_t<scalar_t> toTrade = arrayAccessor.get(mirrorInfo.mirrorGlobalIndex);
        vector<scalar_t, 2> toTradeVector = { toTrade.real(), toTrade.imag() };
        workgroup::Shuffle<shared_memory_adaptor_t, vector<scalar_t, 2> >::__call(toTradeVector, mirrorInfo.otherThreadID, sharedmemAdaptor);
        toTrade.real(toTradeVector.x);
        toTrade.imag(toTradeVector.y);
        return toTrade;
    }

    NBL_CONSTEXPR_STATIC_INLINE indexing_utils_t IndexingUtils;
};




} //namespace fft

// ----------------------------------- End Utils --------------------------------------------------------------

template<bool Inverse, typename consteval_params_t, class device_capabilities=void>
struct FFT;

// For the FFT methods below, we assume:
//      - Accessor is an accessor to an array fitting ElementsPerInvocation * WorkgroupSize elements of type complex_t<Scalar>, used to get inputs / set outputs of the FFT. 
//        If `ConstevalParameters::ElementsPerInvocationLog2 == 1`, the arrays it accesses with `get` and `set` can optionally be different, 
//        if you don't want the FFT to be done in-place. Otherwise, you MUST make it in-place.
//        The Accessor MUST provide the following methods:
//            * void get(uint32_t index, inout complex_t<Scalar> value);
//            * void set(uint32_t index, in complex_t<Scalar> value);
//            * void memoryBarrier();
//        For it to work correctly, this memory barrier must use `AcquireRelease` semantics, with the proper flags set for the memory type.
//        If using `ConstevalParameters::ElementsPerInvocationLog2 == 1` the Accessor IS ALLOWED TO not provide this last method.
//        If not needing it (such as when using preloaded accessors) we still require the method to exist but you can just make it do nothing.
 
//      - SharedMemoryAccessor accesses a workgroup-shared memory array of size `WorkgroupSize` elements of type complex_t<Scalar>.
//        The SharedMemoryAccessor MUST provide the following methods:
//             * void get(uint32_t index, inout uint32_t value);  
//             * void set(uint32_t index, in uint32_t value); 
//             * void workgroupExecutionAndMemoryBarrier();

// 2 items per invocation forward specialization
template<uint16_t WorkgroupSizeLog2, typename Scalar, class device_capabilities>
struct FFT<false, fft::ConstevalParameters<1, WorkgroupSizeLog2, Scalar>, device_capabilities>
{
    using consteval_params_t = fft::ConstevalParameters<1, WorkgroupSizeLog2, Scalar>;

    template<typename SharedMemoryAdaptor>
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        fft::impl::exchangeValues<SharedMemoryAdaptor, Scalar>::__call(lo, hi, threadID, stride, sharedmemAdaptor);
        
        // Get twiddle with k = threadID mod stride, halfN = stride
        hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false, Scalar>(threadID & (stride - 1), stride), lo, hi);    
    }


    template<typename Accessor, typename SharedMemoryAccessor NBL_FUNC_REQUIRES(fft::SmallFFTAccessor<Accessor, Scalar> && fft::FFTSharedMemoryAccessor<SharedMemoryAccessor>)
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = consteval_params_t::WorkgroupSize;

        // Compute the indices only once
        const uint32_t threadID = uint32_t(SubgroupContiguousIndex());
		const uint32_t loIx = threadID;
		const uint32_t hiIx = WorkgroupSize | loIx;

        // Read lo, hi values from global memory
        complex_t<Scalar> lo, hi;
        accessor.get(loIx, lo);
        accessor.get(hiIx, hi);

        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps
        if (WorkgroupSize > glsl::gl_SubgroupSize())
        {
            // Set up the memory adaptor
            using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor,uint32_t,uint32_t,1,WorkgroupSize>;
            adaptor_t sharedmemAdaptor;
            sharedmemAdaptor.accessor = sharedmemAccessor;

            // special first iteration
            hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false, Scalar>(threadID, WorkgroupSize), lo, hi);

            // Run bigger steps until Subgroup-sized
            [unroll]
            for (uint32_t stride = WorkgroupSize >> 1; stride > glsl::gl_SubgroupSize(); stride >>= 1)
            {   
                FFT_loop< adaptor_t >(stride, lo, hi, threadID, sharedmemAdaptor);
                sharedmemAdaptor.workgroupExecutionAndMemoryBarrier(); 
            }

            // special last workgroup-shuffle     
            fft::impl::exchangeValues<adaptor_t, Scalar>::__call(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAdaptor);
            
            // Remember to update the accessor's state
            sharedmemAccessor = sharedmemAdaptor.accessor;
        }

        // Subgroup-sized FFT
        subgroup::FFT<false, Scalar, device_capabilities>::__call(lo, hi);

        // Put values back in global mem
        accessor.set(loIx, lo);
        accessor.set(hiIx, hi);
    }
};

// 2 items per invocation inverse specialization
template<uint16_t WorkgroupSizeLog2, typename Scalar, class device_capabilities>
struct FFT<true, fft::ConstevalParameters<1, WorkgroupSizeLog2, Scalar>, device_capabilities>
{
    using consteval_params_t = fft::ConstevalParameters<1, WorkgroupSizeLog2, Scalar>;

    template<typename SharedMemoryAdaptor>
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        // Get twiddle with k = threadID mod stride, halfN = stride
        hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true, Scalar>(threadID & (stride - 1), stride), lo, hi);     
    
        fft::impl::exchangeValues<SharedMemoryAdaptor, Scalar>::__call(lo, hi, threadID, stride, sharedmemAdaptor);
    }


    template<typename Accessor, typename SharedMemoryAccessor NBL_FUNC_REQUIRES(fft::SmallFFTAccessor<Accessor, Scalar> && fft::FFTSharedMemoryAccessor<SharedMemoryAccessor>)
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = consteval_params_t::WorkgroupSize;

        // Compute the indices only once
        const uint32_t threadID = uint32_t(SubgroupContiguousIndex());
        const uint32_t loIx = threadID;
		const uint32_t hiIx = WorkgroupSize | loIx;

        // Read lo, hi values from global memory
        complex_t<Scalar> lo, hi;
        accessor.get(loIx, lo);
        accessor.get(hiIx, hi);

        // Run a subgroup-sized FFT, then continue with bigger steps
        subgroup::FFT<true, Scalar, device_capabilities>::__call(lo, hi);
        
        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps
        if (WorkgroupSize > glsl::gl_SubgroupSize()) 
        { 
            // Set up the memory adaptor
            using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor,uint32_t,uint32_t,1,WorkgroupSize>;
            adaptor_t sharedmemAdaptor;
            sharedmemAdaptor.accessor = sharedmemAccessor;

            // special first workgroup-shuffle
            fft::impl::exchangeValues<adaptor_t, Scalar>::__call(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAdaptor);
        
            // The bigger steps
            [unroll]
            for (uint32_t stride = glsl::gl_SubgroupSize() << 1; stride < WorkgroupSize; stride <<= 1)
            {   
                // Order of waiting for shared mem writes is also reversed here, since the shuffle came earlier
                sharedmemAdaptor.workgroupExecutionAndMemoryBarrier(); 
                FFT_loop< adaptor_t >(stride, lo, hi, threadID, sharedmemAdaptor);
            }

            // special last iteration 
            hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true, Scalar>(threadID, WorkgroupSize), lo, hi); 
            divides_assign< complex_t<Scalar> > divAss;
            divAss(lo, Scalar(WorkgroupSize / glsl::gl_SubgroupSize()));
            divAss(hi, Scalar(WorkgroupSize / glsl::gl_SubgroupSize()));  

            // Remember to update the accessor's state
            sharedmemAccessor = sharedmemAdaptor.accessor;
        }   
        
        // Put values back in global mem
        accessor.set(loIx, lo);
        accessor.set(hiIx, hi);
    }
};

// Forward FFT
template<uint16_t ElementsPerInvocationLog2, uint16_t WorkgroupSizeLog2, typename Scalar, class device_capabilities>
struct FFT<false, fft::ConstevalParameters<ElementsPerInvocationLog2, WorkgroupSizeLog2, Scalar>, device_capabilities>
{
    using consteval_params_t = fft::ConstevalParameters<ElementsPerInvocationLog2, WorkgroupSizeLog2, Scalar>;
    using small_fft_consteval_params_t = fft::ConstevalParameters<1, WorkgroupSizeLog2, Scalar>;

    template<typename Accessor, typename SharedMemoryAccessor NBL_FUNC_REQUIRES(fft::FFTAccessor<Accessor, Scalar> && fft::FFTSharedMemoryAccessor<SharedMemoryAccessor>)
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = consteval_params_t::WorkgroupSize;
        NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocation = consteval_params_t::ElementsPerInvocation;

        [unroll]
        for (uint32_t stride = (ElementsPerInvocation / 2) * WorkgroupSize; stride > WorkgroupSize; stride >>= 1)
        {
            [unroll]
            for (uint32_t virtualThreadID = SubgroupContiguousIndex(); virtualThreadID < (ElementsPerInvocation / 2) * WorkgroupSize; virtualThreadID += WorkgroupSize)
            {
                const uint32_t loIx = ((virtualThreadID & (~(stride - 1))) << 1) | (virtualThreadID & (stride - 1));
                const uint32_t hiIx = loIx | stride;
                
                complex_t<Scalar> lo, hi;
                accessor.get(loIx, lo);
                accessor.get(hiIx, hi);
                
                hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false,Scalar>(virtualThreadID & (stride - 1), stride),lo,hi);
                
                accessor.set(loIx, lo);
                accessor.set(hiIx, hi);
            }
            accessor.memoryBarrier(); // no execution barrier just making sure writes propagate to accessor
        }

        // do ElementsPerInvocation/2 small workgroup FFTs
        accessor_adaptors::Offset<Accessor> offsetAccessor;
        offsetAccessor.accessor = accessor;
        [unroll]
        for (uint32_t k = 0; k < ElementsPerInvocation; k += 2)
        {
            if (k)
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
            offsetAccessor.offset = WorkgroupSize*k;
            FFT<false, small_fft_consteval_params_t, device_capabilities>::template __call(offsetAccessor,sharedmemAccessor);
        }
        accessor = offsetAccessor.accessor;
    }
};

// Inverse FFT
template<uint16_t ElementsPerInvocationLog2, uint16_t WorkgroupSizeLog2, typename Scalar, class device_capabilities>
struct FFT<true, fft::ConstevalParameters<ElementsPerInvocationLog2, WorkgroupSizeLog2, Scalar>, device_capabilities>
{
    using consteval_params_t = fft::ConstevalParameters<ElementsPerInvocationLog2, WorkgroupSizeLog2, Scalar>;
    using small_fft_consteval_params_t = fft::ConstevalParameters<1, WorkgroupSizeLog2, Scalar>;

    template<typename Accessor, typename SharedMemoryAccessor NBL_FUNC_REQUIRES(fft::FFTAccessor<Accessor, Scalar> && fft::FFTSharedMemoryAccessor<SharedMemoryAccessor>)
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = consteval_params_t::WorkgroupSize;
        NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocation = consteval_params_t::ElementsPerInvocation;

        // do K/2 small workgroup FFTs
        accessor_adaptors::Offset<Accessor> offsetAccessor;
        offsetAccessor.accessor = accessor;
        [unroll]
        for (uint32_t k = 0; k < ElementsPerInvocation; k += 2)
        {
            if (k)
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
            offsetAccessor.offset = WorkgroupSize*k;
            FFT<true, small_fft_consteval_params_t, device_capabilities>::template __call(offsetAccessor,sharedmemAccessor);
        }
        accessor = offsetAccessor.accessor;
        
        [unroll]
        for (uint32_t stride = 2 * WorkgroupSize; stride < ElementsPerInvocation * WorkgroupSize; stride <<= 1)
        {
            accessor.memoryBarrier(); // no execution barrier just making sure writes propagate to accessor
            [unroll]
            for (uint32_t virtualThreadID = SubgroupContiguousIndex(); virtualThreadID < (ElementsPerInvocation / 2) * WorkgroupSize; virtualThreadID += WorkgroupSize)
            {
                const uint32_t loIx = ((virtualThreadID & (~(stride - 1))) << 1) | (virtualThreadID & (stride - 1));
                const uint32_t hiIx = loIx | stride;
                
                complex_t<Scalar> lo, hi;
                accessor.get(loIx, lo);
                accessor.get(hiIx, hi);

                hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true,Scalar>(virtualThreadID & (stride - 1), stride), lo,hi);
                
                // Divide by special factor at the end
                if ( (ElementsPerInvocation / 2) * WorkgroupSize == stride)
                {
                    divides_assign< complex_t<Scalar> > divAss;
                    divAss(lo, ElementsPerInvocation / 2);
                    divAss(hi, ElementsPerInvocation / 2);  
                }

                accessor.set(loIx, lo);
                accessor.set(hiIx, hi);
            }
           
        }
        
    }
};

}
}
}

// ------------------------------- END HLSL ONLY ----------------------------------------------
#endif

#endif