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
        // Ternary won't take structs so we use this aux variable
        uint32_t toExchange = bit_cast<uint32_t, vector <float16_t, 2> >(topHalf ? vector <float16_t, 2> (lo.real(), lo.imag()) : vector <float16_t, 2> (hi.real(), hi.imag()));
        shuffleXor<SharedMemoryAdaptor, uint32_t>::__call(toExchange, stride, sharedmemAdaptor);
        vector <float16_t, 2> exchanged = bit_cast<vector <float16_t, 2>, uint32_t>(toExchange);
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
        // Ternary won't take structs so we use this aux variable
        vector <uint32_t, 2> toExchange = bit_cast<vector <uint32_t, 2>, vector <float32_t, 2> >(topHalf ? vector <float32_t, 2>(lo.real(), lo.imag()) : vector <float32_t, 2>(hi.real(), hi.imag()));
        shuffleXor<SharedMemoryAdaptor, vector <uint32_t, 2> >::__call(toExchange, stride, sharedmemAdaptor);
        vector <float32_t, 2> exchanged = bit_cast<vector <float32_t, 2>, vector <uint32_t, 2> >(toExchange);
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
        // Ternary won't take structs so we use this aux variable
        vector <uint32_t, 4> toExchange = bit_cast<vector <uint32_t, 4>, vector <float64_t, 2> > (topHalf ? vector <float64_t, 2>(lo.real(), lo.imag()) : vector <float64_t, 2>(hi.real(), hi.imag()));                       
        shuffleXor<SharedMemoryAdaptor, vector <uint32_t, 4> >::__call(toExchange, stride, sharedmemAdaptor);
        vector <float64_t, 2> exchanged = bit_cast<vector <float64_t, 2>, vector <uint32_t, 4> >(toExchange);
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
//        that is, one "lo" and one "hi" complex numbers per thread, essentially 4 Scalars per thread. 
//        There are no assumptions on the data layout: we just require the accessor to provide get and set methods for complex_t<Scalar>.
//      - SharedMemoryAccessor accesses a shared memory array that can fit _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar>, with get and set 
//        methods for complex_t<Scalar>. It benefits from coalesced accesses   

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
            MemoryAdaptor<SharedMemoryAccessor> sharedmemAdaptor;
            sharedmemAdaptor.accessor = sharedmemAccessor;

            // special first iteration
            hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false, Scalar>(threadID, _NBL_HLSL_WORKGROUP_SIZE_), lo, hi);

            // Run bigger steps until Subgroup-sized
            for (uint32_t stride = _NBL_HLSL_WORKGROUP_SIZE_ >> 1; stride > glsl::gl_SubgroupSize(); stride >>= 1)
            {   
                FFT_loop< MemoryAdaptor<SharedMemoryAccessor> >(stride, lo, hi, threadID, sharedmemAdaptor);
                sharedmemAdaptor.workgroupExecutionAndMemoryBarrier(); 
            }

            // special last workgroup-shuffle     
            fft::exchangeValues<MemoryAdaptor<SharedMemoryAccessor>, Scalar>::__call(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAdaptor);  
            
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
            MemoryAdaptor<SharedMemoryAccessor> sharedmemAdaptor;
            sharedmemAdaptor.accessor = sharedmemAccessor;

            // special first workgroup-shuffle
            fft::exchangeValues<MemoryAdaptor<SharedMemoryAccessor>, Scalar>::__call(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAdaptor);
        
            // The bigger steps
            for (uint32_t stride = glsl::gl_SubgroupSize() << 1; stride < _NBL_HLSL_WORKGROUP_SIZE_; stride <<= 1)
            {   
                // Order of waiting for shared mem writes is also reversed here, since the shuffle came earlier
                sharedmemAdaptor.workgroupExecutionAndMemoryBarrier(); 
                FFT_loop< MemoryAdaptor<SharedMemoryAccessor> >(stride, lo, hi, threadID, sharedmemAdaptor);
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
        for (uint32_t stride = (K / 2) * _NBL_HLSL_WORKGROUP_SIZE_; stride > _NBL_HLSL_WORKGROUP_SIZE_; stride >>= 1)
        {
            //[unroll(K/2)]
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
        DynamicOffsetAccessor <Accessor> offsetAccessor;
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
        DynamicOffsetAccessor <Accessor> offsetAccessor;
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