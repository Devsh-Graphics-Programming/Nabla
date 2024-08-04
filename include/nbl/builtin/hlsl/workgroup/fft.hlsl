#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/workgroup/shuffle.hlsl"
#include "nbl/builtin/hlsl/mpl.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{
namespace fft
{

// ---------------------------------- Utils -----------------------------------------------

template<typename SharedMemoryAccessor, typename Scalar>
void exchangeValues(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, uint32_t stride, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
{
    const bool topHalf = bool(threadID & stride);
    // Ternary won't take structs so we use this aux variable
    vector <Scalar, 2> toExchange = topHalf ? vector <Scalar, 2>(lo.real(), lo.imag()) : vector <Scalar, 2>(hi.real(), hi.imag());
    complex_t<Scalar> toExchangeComplex = {toExchange.x, toExchange.y};
    shuffleXor<SharedMemoryAccessor, complex_t<Scalar> >::__call(toExchangeComplex, stride, sharedmemAccessor);
    if (topHalf)
        lo = toExchangeComplex;
    else
        hi = toExchangeComplex;
}

} //namespace fft

// ----------------------------------- End Utils -----------------------------------------------

template<uint16_t ElementsPerInvocation, bool Inverse, typename Scalar, class device_capabilities=void>
struct FFT;

// For the FFT methods below, we assume:
//      - Accessor is a global memory accessor to an array fitting 2 * _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar>, used to get inputs / set outputs of the FFT,
//        that is, one "lo" and one "hi" complex numbers per thread, essentially 4 Scalars per thread. 
//        There are no assumptions on the data layout: we just require the accessor to provide get and set methods for complex_t<Scalar>.
//      - SharedMemoryAccessor accesses a shared memory array that can fit _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar>, with get and set 
//        methods for complex_t<Scalar>. It benefits from coalesced accesses   

// 2 items per invocation forward specialization
template<typename Scalar, class device_capabilities>
struct FFT<2,false, Scalar, device_capabilities>
{
    template<typename SharedMemoryAccessor>
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        fft::exchangeValues<SharedMemoryAccessor, Scalar>(lo, hi, threadID, stride, sharedmemAccessor);
        
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
        complex_t<Scalar> lo = accessor.get(loIx);
        complex_t<Scalar> hi = accessor.get(hiIx);

        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps
        if (_NBL_HLSL_WORKGROUP_SIZE_ > glsl::gl_SubgroupSize())
        {
            // special first iteration
            hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false, Scalar>(threadID, _NBL_HLSL_WORKGROUP_SIZE_), lo, hi);

            // Run bigger steps until Subgroup-sized
            for (uint32_t stride = _NBL_HLSL_WORKGROUP_SIZE_ >> 1; stride > glsl::gl_SubgroupSize(); stride >>= 1)
            {   
                FFT_loop<SharedMemoryAccessor>(stride, lo, hi, threadID, sharedmemAccessor);
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier(); 
            }

            // special last workgroup-shuffle     
            fft::exchangeValues<SharedMemoryAccessor, Scalar>(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAccessor);  
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
    template<typename SharedMemoryAccessor>
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        // Get twiddle with k = threadID mod stride, halfN = stride
        hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true, Scalar>(threadID & (stride - 1), stride), lo, hi);     
    
        fft::exchangeValues<SharedMemoryAccessor, Scalar>(lo, hi, threadID, stride, sharedmemAccessor);
    }


    template<typename Accessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        // Compute the indices only once
        const uint32_t threadID = uint32_t(SubgroupContiguousIndex());
        const uint32_t loIx = threadID;
		const uint32_t hiIx = _NBL_HLSL_WORKGROUP_SIZE_ | loIx;

        // Read lo, hi values from global memory
        complex_t<Scalar> lo = accessor.get(loIx);
        complex_t<Scalar> hi = accessor.get(hiIx);

        // Run a subgroup-sized FFT, then continue with bigger steps
        subgroup::FFT<true, Scalar, device_capabilities>::__call(lo, hi);
        
        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps
        if (_NBL_HLSL_WORKGROUP_SIZE_ > glsl::gl_SubgroupSize()) 
        { 
            // special first workgroup-shuffle
            fft::exchangeValues<SharedMemoryAccessor, Scalar>(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAccessor);
        
            // The bigger steps
            for (uint32_t stride = glsl::gl_SubgroupSize() << 1; stride < _NBL_HLSL_WORKGROUP_SIZE_; stride <<= 1)
            {   
                // Order of waiting for shared mem writes is also reversed here, since the shuffle came earlier
                sharedmemAccessor.workgroupExecutionAndMemoryBarrier(); 
                FFT_loop<SharedMemoryAccessor>(stride, lo, hi, threadID, sharedmemAccessor);
            }

            // special last iteration 
            hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true, Scalar>(threadID, _NBL_HLSL_WORKGROUP_SIZE_), lo, hi); 
            divides_assign< complex_t<Scalar> > divAss;
            divAss(lo, _NBL_HLSL_WORKGROUP_SIZE_ / glsl::gl_SubgroupSize());
            divAss(hi, _NBL_HLSL_WORKGROUP_SIZE_ / glsl::gl_SubgroupSize());      
        }   
        
        // Put values back in global mem
        accessor.set(loIx, lo);
        accessor.set(hiIx, hi);
    }
};


// ---------------------------- Below pending --------------------------------------------------

// Forward FFT
template<uint32_t K, typename Scalar, class device_capabilities>
struct FFT<K, false, Scalar, device_capabilities>
{
    template<typename Accessor, typename SharedMemoryAccessor>
    static enable_if_t< (mpl::is_pot_v<K> && K > 2), void > __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        for (uint32_t stride = (K >> 1) * _NBL_HLSL_WORKGROUP_SIZE_; stride > _NBL_HLSL_WORKGROUP_SIZE_; stride >>= 1)
        {
            //[unroll(K/2)]
            for (uint32_t virtualThreadID = SubgroupContiguousIndex(); virtualThreadID < (K >> 1) * _NBL_HLSL_WORKGROUP_SIZE_; virtualThreadID += _NBL_HLSL_WORKGROUP_SIZE_)
            {
                const uint32_t loIx = ((virtualThreadID & (~(stride - 1))) << 1) | (virtualThreadID & (stride - 1));
                const uint32_t hiIx = loIx | stride;
                
                complex_t<Scalar> lo = accessor.get(loIx);
                complex_t<Scalar> hi = accessor.get(hiIx);
                
                hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false,Scalar>(virtualThreadID & (stride - 1), stride),lo,hi);
                
                accessor.set(loIx, lo);
                accessor.set(hiIx, hi);
            }
            accessor.memoryBarrier(); // no execution barrier just making sure writes propagate to accessor
        }

        // do K/2 small workgroup FFTs
        DynamicOffsetAccessor < Accessor, complex_t<Scalar> > offsetAccessor;
        //[unroll(K/2)]
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
        DynamicOffsetAccessor < Accessor, complex_t<Scalar> > offsetAccessor;
        //[unroll(K/2)]
        for (uint32_t k = 0; k < K; k += 2)
        {
            if (k)
            sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
            offsetAccessor.offset = _NBL_HLSL_WORKGROUP_SIZE_*k;
            FFT<2,true, Scalar, device_capabilities>::template __call(offsetAccessor,sharedmemAccessor);
        }
        accessor = offsetAccessor.accessor;
      
        for (uint32_t stride = 2 * _NBL_HLSL_WORKGROUP_SIZE_; stride < K * _NBL_HLSL_WORKGROUP_SIZE_; stride <<= 1)
        {
            accessor.memoryBarrier(); // no execution barrier just making sure writes propagate to accessor
            //[unroll(K/2)]
            for (uint32_t virtualThreadID = SubgroupContiguousIndex(); virtualThreadID < (K >> 1) * _NBL_HLSL_WORKGROUP_SIZE_; virtualThreadID += _NBL_HLSL_WORKGROUP_SIZE_)
            {
                const uint32_t loIx = ((virtualThreadID & (~(stride - 1))) << 1) | (virtualThreadID & (stride - 1));
                const uint32_t hiIx = loIx | stride;
                
                complex_t<Scalar> lo = accessor.get(loIx);
                complex_t<Scalar> hi = accessor.get(hiIx);

                hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true,Scalar>(virtualThreadID & (stride - 1), stride), lo,hi);
                
                // Divide by special factor at the end
                if ( (K >> 1) * _NBL_HLSL_WORKGROUP_SIZE_ == stride)
                {
                    divides_assign< complex_t<Scalar> > divAss;
                    divAss(lo, K >> 1);
                    divAss(hi, K >> 1);  
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