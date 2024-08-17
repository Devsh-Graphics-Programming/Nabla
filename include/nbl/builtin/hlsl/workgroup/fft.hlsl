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
struct exchangeValues;

template<typename SharedMemoryAdaptor>
struct exchangeValues<SharedMemoryAdaptor, float16_t>
{
    static void __call(NBL_REF_ARG(complex_t<float16_t>) lo, NBL_REF_ARG(complex_t<float16_t>) hi, uint32_t threadID, uint32_t stride, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        const bool topHalf = bool(threadID & stride);
        // Pack two halves into a single uint32_t
        uint32_t toExchange = bit_cast<uint32_t, float16_t2 >(topHalf ? float16_t2 (lo.real(), lo.imag()) : float16_t2 (hi.real(), hi.imag()));
        shuffleXor<SharedMemoryAdaptor, uint32_t>::__call(toExchange, stride, sharedmemAdaptor);
        float16_t2 exchanged = bit_cast<float16_t2, uint32_t>(toExchange);
        if (topHalf)
        {
            lo.real(exchanged.x);
            lo.imag(exchanged.y);
        }
        else
        {
            hi.real(exchanged.x);
            lo.imag(exchanged.y);
        }   
    }
};

template<typename SharedMemoryAdaptor>
struct exchangeValues<SharedMemoryAdaptor, float32_t>
{
    static void __call(NBL_REF_ARG(complex_t<float32_t>) lo, NBL_REF_ARG(complex_t<float32_t>) hi, uint32_t threadID, uint32_t stride, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        const bool topHalf = bool(threadID & stride);
        // pack into `float32_t2` because ternary operator doesn't support structs
        float32_t2 exchanged = topHalf ? float32_t2(lo.real(), lo.imag()) : float32_t2(hi.real(), hi.imag());
        shuffleXor<SharedMemoryAdaptor, float32_t2>::__call(exchanged, stride, sharedmemAdaptor);
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

template<typename SharedMemoryAdaptor>
struct exchangeValues<SharedMemoryAdaptor, float64_t>
{
    static void __call(NBL_REF_ARG(complex_t<float64_t>) lo, NBL_REF_ARG(complex_t<float64_t>) hi, uint32_t threadID, uint32_t stride, NBL_REF_ARG(SharedMemoryAdaptor) sharedmemAdaptor)
    {
        const bool topHalf = bool(threadID & stride);
        // pack into `float64_t2` because ternary operator doesn't support structs
        float64_t2 exchanged = topHalf ? float64_t2(lo.real(), lo.imag()) : float64_t2(hi.real(), hi.imag());                    
        shuffleXor<SharedMemoryAdaptor, float64_t2 >::__call(exchanged, stride, sharedmemAdaptor);
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

} //namespace fft

// ----------------------------------- End Utils -----------------------------------------------

template<uint16_t ElementsPerInvocation, bool Inverse, typename Scalar, class device_capabilities=void>
struct FFT;

// For the FFT methods below, we assume:
//      - Accessor is a global memory accessor to an array fitting 2 * _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar>, used to get inputs / set outputs of the FFT,
//        that is, one "lo" and one "hi" complex numbers per thread, essentially 4 Scalars per thread. The arrays it accesses with `get` and `set` can optionally be
//        different, if you don't want the FFT to be done in-place. 
//        The Accessor MUST provide the following methods:
//            * void get(uint32_t index, inout complex_t<Scalar> value);
//            * void set(uint32_t index, in complex_t<Scalar> value);
//            * void memoryBarrier();
//        You might optionally want to provide a `workgroupExecutionAndMemoryBarrier()` method on it to wait on to be sure the whole FFT pass is done
 
//      - SharedMemoryAccessor accesses a workgroup-shared memory array of size `2 * sizeof(Scalar) * _NBL_HLSL_WORKGROUP_SIZE_`.
//        The SharedMemoryAccessor MUST provide the following methods:
//             * void get(uint32_t index, inout uint32_t value);  
//             * void set(uint32_t index, in uint32_t value); 
//             * void workgroupExecutionAndMemoryBarrier();

// 2 items per invocation forward specialization
template<typename Scalar, class device_capabilities>
struct FFT<2,false, Scalar, device_capabilities>
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
		const uint32_t hiIx = _NBL_HLSL_WORKGROUP_SIZE_ | loIx;

        // Read lo, hi values from global memory
        complex_t<Scalar> lo, hi;
        accessor.get(loIx, lo);
        accessor.get(hiIx, hi);

        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps
        if (_NBL_HLSL_WORKGROUP_SIZE_ > glsl::gl_SubgroupSize())
        {
            // Set up the memory adaptor
            using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor,uint32_t,uint32_t,1,_NBL_HLSL_WORKGROUP_SIZE_>;
            adaptor_t sharedmemAdaptor;
            sharedmemAdaptor.accessor = sharedmemAccessor;

            // special first iteration
            hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false, Scalar>(threadID, _NBL_HLSL_WORKGROUP_SIZE_), lo, hi);

            // Run bigger steps until Subgroup-sized
            [unroll]
            for (uint32_t stride = _NBL_HLSL_WORKGROUP_SIZE_ >> 1; stride > glsl::gl_SubgroupSize(); stride >>= 1)
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
template<typename Scalar, class device_capabilities>
struct FFT<2,true, Scalar, device_capabilities>
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
		const uint32_t hiIx = _NBL_HLSL_WORKGROUP_SIZE_ | loIx;

        // Read lo, hi values from global memory
        complex_t<Scalar> lo, hi;
        accessor.get(loIx, lo);
        accessor.get(hiIx, hi);

        // Run a subgroup-sized FFT, then continue with bigger steps
        subgroup::FFT<true, Scalar, device_capabilities>::__call(lo, hi);
        
        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps
        if (_NBL_HLSL_WORKGROUP_SIZE_ > glsl::gl_SubgroupSize()) 
        { 
            // Set up the memory adaptor
            using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor,uint32_t,uint32_t,1,_NBL_HLSL_WORKGROUP_SIZE_>;
            adaptor_t sharedmemAdaptor;
            sharedmemAdaptor.accessor = sharedmemAccessor;

            // special first workgroup-shuffle
            fft::exchangeValues<adaptor_t, Scalar>::__call(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAdaptor);
        
            // The bigger steps
            [unroll]
            for (uint32_t stride = glsl::gl_SubgroupSize() << 1; stride < _NBL_HLSL_WORKGROUP_SIZE_; stride <<= 1)
            {   
                // Order of waiting for shared mem writes is also reversed here, since the shuffle came earlier
                sharedmemAdaptor.workgroupExecutionAndMemoryBarrier(); 
                FFT_loop< adaptor_t >(stride, lo, hi, threadID, sharedmemAdaptor);
            }

            // special last iteration 
            hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true, Scalar>(threadID, _NBL_HLSL_WORKGROUP_SIZE_), lo, hi); 
            divides_assign< complex_t<Scalar> > divAss;
            divAss(lo, Scalar(_NBL_HLSL_WORKGROUP_SIZE_ / glsl::gl_SubgroupSize()));
            divAss(hi, Scalar(_NBL_HLSL_WORKGROUP_SIZE_ / glsl::gl_SubgroupSize()));  

            // Remember to update the accessor's state
            sharedmemAccessor = sharedmemAdaptor.accessor;
        }   
        
        // Put values back in global mem
        accessor.set(loIx, lo);
        accessor.set(hiIx, hi);
    }
};

// Forward FFT
template<uint32_t K, typename Scalar, class device_capabilities>
struct FFT<K, false, Scalar, device_capabilities>
{
    template<typename Accessor, typename SharedMemoryAccessor>
    static enable_if_t< (mpl::is_pot_v<K> && K > 2), void > __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        [unroll]
        for (uint32_t stride = (K / 2) * _NBL_HLSL_WORKGROUP_SIZE_; stride > _NBL_HLSL_WORKGROUP_SIZE_; stride >>= 1)
        {
            [unroll]
            for (uint32_t virtualThreadID = SubgroupContiguousIndex(); virtualThreadID < (K / 2) * _NBL_HLSL_WORKGROUP_SIZE_; virtualThreadID += _NBL_HLSL_WORKGROUP_SIZE_)
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
        [unroll]
        for (uint32_t k = 0; k < K; k += 2)
        {
            if (k)
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
            offsetAccessor.offset = _NBL_HLSL_WORKGROUP_SIZE_*k;
            FFT<2,false, Scalar, device_capabilities>::template __call(offsetAccessor,sharedmemAccessor);
        }
        accessor = offsetAccessor.accessor;
    }
};

// Inverse FFT
template<uint32_t K, typename Scalar, class device_capabilities>
struct FFT<K, true, Scalar, device_capabilities>
{
    template<typename Accessor, typename SharedMemoryAccessor>
    static enable_if_t< (mpl::is_pot_v<K> && K > 2), void > __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        // do K/2 small workgroup FFTs
        accessor_adaptors::Offset<Accessor> offsetAccessor;
        [unroll]
        for (uint32_t k = 0; k < K; k += 2)
        {
            if (k)
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
            offsetAccessor.offset = _NBL_HLSL_WORKGROUP_SIZE_*k;
            FFT<2,true, Scalar, device_capabilities>::template __call(offsetAccessor,sharedmemAccessor);
        }
        accessor = offsetAccessor.accessor;
        
        [unroll]
        for (uint32_t stride = 2 * _NBL_HLSL_WORKGROUP_SIZE_; stride < K * _NBL_HLSL_WORKGROUP_SIZE_; stride <<= 1)
        {
            accessor.memoryBarrier(); // no execution barrier just making sure writes propagate to accessor
            [unroll]
            for (uint32_t virtualThreadID = SubgroupContiguousIndex(); virtualThreadID < (K / 2) * _NBL_HLSL_WORKGROUP_SIZE_; virtualThreadID += _NBL_HLSL_WORKGROUP_SIZE_)
            {
                const uint32_t loIx = ((virtualThreadID & (~(stride - 1))) << 1) | (virtualThreadID & (stride - 1));
                const uint32_t hiIx = loIx | stride;
                
                complex_t<Scalar> lo, hi;
                accessor.get(loIx, lo);
                accessor.get(hiIx, hi);

                hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true,Scalar>(virtualThreadID & (stride - 1), stride), lo,hi);
                
                // Divide by special factor at the end
                if ( (K / 2) * _NBL_HLSL_WORKGROUP_SIZE_ == stride)
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