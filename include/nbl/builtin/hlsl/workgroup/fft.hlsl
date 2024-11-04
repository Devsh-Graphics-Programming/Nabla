#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/shuffle.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"

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

// Get the required size (in number of uint32_t elements) of the workgroup shared memory array needed for the FFT
template <typename scalar_t, uint32_t WorkgroupSize>
NBL_CONSTEXPR uint32_t SharedMemoryDWORDs = (sizeof(complex_t<scalar_t>) / sizeof(uint32_t))  * WorkgroupSize;


template<uint32_t N, uint32_t H>
enable_if_t<H <= N, uint32_t> bitShiftRightHigher(uint32_t i)
{
    // Highest H bits are numbered N-1 through N - H
    // N - H is then the middle bit
    // Lowest bits numbered from 0 through N - H - 1
    uint32_t low = i & ((1 << (N - H)) - 1);
    uint32_t mid = i & (1 << (N - H));
    uint32_t high = i & ~((1 << (N - H + 1)) - 1);

    high >>= 1;
    mid <<= H - 1;

    return mid | high | low;
}

template<uint32_t N, uint32_t H>
enable_if_t<H <= N, uint32_t> bitShiftLeftHigher(uint32_t i)
{
    // Highest H bits are numbered N-1 through N - H
    // N - 1 is then the highest bit, and N - 2 through N - H are the middle bits
    // Lowest bits numbered from 0 through N - H - 1
    uint32_t low = i & ((1 << (N - H)) - 1);
    uint32_t mid = i & (~((1 << (N - H)) - 1) | ~(1 << (N - 1)));
    uint32_t high = i & (1 << (N - 1));

    mid <<= 1;
    high >>= H - 1;

    return mid | high | low;
}

// This function maps the index `idx` in the output array of a Forward FFT to the index `freqIdx` in the DFT such that `DFT[freqIdx] = output[idx]`
// This is because Cooley-Tukey + subgroup operations end up spewing out the outputs in a weird order
template<uint16_t ElementsPerInvocation, uint32_t WorkgroupSize>
uint32_t getFrequencyIndex(uint32_t outputIdx)
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ELEMENTS_PER_INVOCATION_LOG_2 = uint32_t(mpl::log2<ElementsPerInvocation>::value);
    NBL_CONSTEXPR_STATIC_INLINE uint32_t FFT_SIZE_LOG_2 = ELEMENTS_PER_INVOCATION_LOG_2 + uint32_t(mpl::log2<WorkgroupSize>::value);

    return bitShiftRightHigher<FFT_SIZE_LOG_2, FFT_SIZE_LOG_2 - ELEMENTS_PER_INVOCATION_LOG_2 + 1>(glsl::bitfieldReverse<uint32_t>(outputIdx) >> (32 - FFT_SIZE_LOG_2));
}

// This function maps the index `freqIdx` in the DFT to the index `idx` in the output array of a Forward FFT such that `DFT[freqIdx] = output[idx]`
// It is essentially the inverse of `getFrequencyIndex`
template<uint16_t ElementsPerInvocation, uint32_t WorkgroupSize>
uint32_t getOutputIndex(uint32_t freqIdx)
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ELEMENTS_PER_INVOCATION_LOG_2 = uint32_t(mpl::log2<ElementsPerInvocation>::value);
    NBL_CONSTEXPR_STATIC_INLINE uint32_t FFT_SIZE_LOG_2 = ELEMENTS_PER_INVOCATION_LOG_2 + uint32_t(mpl::log2<WorkgroupSize>::value);

    return glsl::bitfieldReverse<uint32_t>(bitShiftLeftHigher<FFT_SIZE_LOG_2, FFT_SIZE_LOG_2 - ELEMENTS_PER_INVOCATION_LOG_2 + 1>(freqIdx)) >> (32 - FFT_SIZE_LOG_2);
}

// Mirrors an index about the Nyquist frequency
template<uint16_t ElementsPerInvocation, uint32_t WorkgroupSize>
uint32_t mirror(uint32_t idx)
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t FFT_SIZE = WorkgroupSize * uint32_t(ElementsPerInvocation);
    return (FFT_SIZE - idx) & (FFT_SIZE - 1);
}

// When packing real FFTs a common operation is to get `DFT[T]` and `DFT[-T]` to unpack the result of a packed real FFT.
// Given an index `idx` into the Nabla-ordered DFT such that `output[idx] = DFT[T]`, this function is such that `output[getNegativeIndex(idx)] = DFT[-T]`
template<uint16_t ElementsPerInvocation, uint32_t WorkgroupSize>
uint32_t getNegativeIndex(uint32_t idx)
{
    return getOutputIndex<ElementsPerInvocation, WorkgroupSize>(mirror<ElementsPerInvocation, WorkgroupSize>(getFrequencyIndex<ElementsPerInvocation, WorkgroupSize>(idx)));
}

// Util to unpack two values from the packed FFT X + iY - get outputs in the same input arguments, storing x to lo and y to hi
template<typename Scalar>
void unpack(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi)
{
    complex_t<Scalar> x = (lo + conj(hi)) * Scalar(0.5);
    hi = rotateRight<Scalar>(lo - conj(hi)) * Scalar(0.5);
    lo = x;
}

} //namespace fft

// ----------------------------------- End Utils -----------------------------------------------

template<uint16_t ElementsPerInvocation, bool Inverse, uint32_t WorkgroupSize, typename Scalar, class device_capabilities=void>
struct FFT;

// For the FFT methods below, we assume:
//      - Accessor is a global memory accessor to an array fitting 2 * WorkgroupSize elements of type complex_t<Scalar>, used to get inputs / set outputs of the FFT,
//        that is, one "lo" and one "hi" complex numbers per thread, essentially 4 Scalars per thread. The arrays it accesses with `get` and `set` can optionally be
//        different, if you don't want the FFT to be done in-place. 
//        The Accessor MUST provide the following methods:
//            * void get(uint32_t index, inout complex_t<Scalar> value);
//            * void set(uint32_t index, in complex_t<Scalar> value);
//            * void memoryBarrier();
//        You might optionally want to provide a `workgroupExecutionAndMemoryBarrier()` method on it to wait on to be sure the whole FFT pass is done
 
//      - SharedMemoryAccessor accesses a workgroup-shared memory array of size `2 * sizeof(Scalar) * WorkgroupSize`.
//        The SharedMemoryAccessor MUST provide the following methods:
//             * void get(uint32_t index, inout uint32_t value);  
//             * void set(uint32_t index, in uint32_t value); 
//             * void workgroupExecutionAndMemoryBarrier();

// 2 items per invocation forward specialization
template<uint32_t WorkgroupSize, typename Scalar, class device_capabilities>
struct FFT<2,false, WorkgroupSize, Scalar, device_capabilities>
{
    template<typename SharedMemoryAdaptor>
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        fft::exchangeValues<SharedMemoryAdaptor, Scalar>::__call(lo, hi, threadID, stride, sharedmemAdaptor);
        
        // Get twiddle with k = threadID mod stride, halfN = stride
        hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false, Scalar>(threadID & (stride - 1), stride), lo, hi);    
    }


    template<typename Accessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
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
            fft::exchangeValues<adaptor_t, Scalar>::__call(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAdaptor);
            
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
template<uint32_t WorkgroupSize, typename Scalar, class device_capabilities>
struct FFT<2,true, WorkgroupSize, Scalar, device_capabilities>
{
    template<typename SharedMemoryAdaptor>
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        // Get twiddle with k = threadID mod stride, halfN = stride
        hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true, Scalar>(threadID & (stride - 1), stride), lo, hi);     
    
        fft::exchangeValues<SharedMemoryAdaptor, Scalar>::__call(lo, hi, threadID, stride, sharedmemAdaptor);
    }


    template<typename Accessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
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
            fft::exchangeValues<adaptor_t, Scalar>::__call(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAdaptor);
        
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
template<uint32_t K, uint32_t WorkgroupSize, typename Scalar, class device_capabilities>
struct FFT<K, false, WorkgroupSize, Scalar, device_capabilities>
{
    template<typename Accessor, typename SharedMemoryAccessor>
    static enable_if_t< (mpl::is_pot_v<K> && K > 2), void > __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        [unroll]
        for (uint32_t stride = (K / 2) * WorkgroupSize; stride > WorkgroupSize; stride >>= 1)
        {
            [unroll]
            for (uint32_t virtualThreadID = SubgroupContiguousIndex(); virtualThreadID < (K / 2) * WorkgroupSize; virtualThreadID += WorkgroupSize)
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

        // do K/2 small workgroup FFTs
        accessor_adaptors::Offset<Accessor> offsetAccessor;
        offsetAccessor.accessor = accessor;
        [unroll]
        for (uint32_t k = 0; k < K; k += 2)
        {
            if (k)
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
            offsetAccessor.offset = WorkgroupSize*k;
            FFT<2,false, WorkgroupSize, Scalar, device_capabilities>::template __call(offsetAccessor,sharedmemAccessor);
        }
        accessor = offsetAccessor.accessor;
    }
};

// Inverse FFT
template<uint32_t K, uint32_t WorkgroupSize, typename Scalar, class device_capabilities>
struct FFT<K, true, WorkgroupSize, Scalar, device_capabilities>
{
    template<typename Accessor, typename SharedMemoryAccessor>
    static enable_if_t< (mpl::is_pot_v<K> && K > 2), void > __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        // do K/2 small workgroup FFTs
        accessor_adaptors::Offset<Accessor> offsetAccessor;
        offsetAccessor.accessor = accessor;
        [unroll]
        for (uint32_t k = 0; k < K; k += 2)
        {
            if (k)
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
            offsetAccessor.offset = WorkgroupSize*k;
            FFT<2,true, WorkgroupSize, Scalar, device_capabilities>::template __call(offsetAccessor,sharedmemAccessor);
        }
        accessor = offsetAccessor.accessor;
        
        [unroll]
        for (uint32_t stride = 2 * WorkgroupSize; stride < K * WorkgroupSize; stride <<= 1)
        {
            accessor.memoryBarrier(); // no execution barrier just making sure writes propagate to accessor
            [unroll]
            for (uint32_t virtualThreadID = SubgroupContiguousIndex(); virtualThreadID < (K / 2) * WorkgroupSize; virtualThreadID += WorkgroupSize)
            {
                const uint32_t loIx = ((virtualThreadID & (~(stride - 1))) << 1) | (virtualThreadID & (stride - 1));
                const uint32_t hiIx = loIx | stride;
                
                complex_t<Scalar> lo, hi;
                accessor.get(loIx, lo);
                accessor.get(hiIx, hi);

                hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true,Scalar>(virtualThreadID & (stride - 1), stride), lo,hi);
                
                // Divide by special factor at the end
                if ( (K / 2) * WorkgroupSize == stride)
                {
                    divides_assign< complex_t<Scalar> > divAss;
                    divAss(lo, K / 2);
                    divAss(hi, K / 2);  
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

#endif