#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/workgroup/shuffle.hlsl"

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
void exchangeValues(NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, uint32_t stride, NBL_REF_ARG(MemoryAdaptor<SharedMemoryAccessor>) sharedmemAdaptor)
{
    const bool topHalf = bool(threadID & stride);
    vector <Scalar, 2> toExchange = topHalf ? vector <Scalar, 2>(lo.real(), lo.imag()) : vector <Scalar, 2>(hi.real(), hi.imag());
    shuffleXor<SharedMemoryAccessor, vector <Scalar, 2> >::__call(toExchange, stride, sharedmemAdaptor);
    if (topHalf)
    {
        lo.real(toExchange.x);
        lo.imag(toExchange.y);
    }
    else
    {
        hi.real(toExchange.x);
        hi.imag(toExchange.y);
    }   
}

// ----------------------------------- End Utils -----------------------------------------------

template<uint16_t ElementsPerInvocation, bool Inverse, typename Scalar, class device_capabilities=void>
struct FFT;

// For the FFT methods below, we assume:
//      - Accessor is a global memory accessor to an array fitting 2 * _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar>, used to get inputs / set outputs of the FFT,
//        that is, one "lo" and one "hi" complex numbers per thread, essentially 4 Scalars per thread. The data layout is assumed to be a whole array of real parts 
//        followed by a whole array of imaginary parts. So it would be something like
//        [x_0, x_1, ..., x_{2 * _NBL_HLSL_WORKGROUP_SIZE_}, y_0, y_1, ..., y_{2 * _NBL_HLSL_WORKGROUP_SIZE_}] 
//      - SharedMemoryAccessor accesses a shared memory array that can fit _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar>, so 2 * _NBL_HLSL_WORKGROUP_SIZE_ Scalars 

// 2 items per invocation forward specialization
template<typename Scalar, class device_capabilities>
struct FFT<2,false, Scalar, device_capabilities>
{
    template<typename SharedMemoryAccessor>
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, NBL_REF_ARG(MemoryAdaptor<SharedMemoryAccessor>) sharedmemAdaptor)
    {
        exchangeValues<SharedMemoryAccessor, Scalar>(lo, hi, threadID, stride, sharedmemAdaptor);
        
        // Get twiddle with k = threadID mod stride, halfN = stride
        hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false, Scalar>(threadID & (stride - 1), stride), lo, hi);    
    }


    template<typename Accessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        // Set up the MemAdaptors
        MemoryAdaptor<Accessor, 1> memAdaptor;
        memAdaptor.accessor = accessor;
        MemoryAdaptor<SharedMemoryAccessor> sharedmemAdaptor;
        sharedmemAdaptor.accessor = sharedmemAccessor;

        // Compute the indices only once
        const uint32_t threadID = uint32_t(SubgroupContiguousIndex());
		const uint32_t loIx = threadID;
		const uint32_t hiIx = loIx + _NBL_HLSL_WORKGROUP_SIZE_;

        // Read lo, hi values from global memory
        vector <Scalar, 2> loVec;
        vector <Scalar, 2> hiVec;
        // TODO: if we get rid of the Memory Adaptor on the accessor and require comples getters and setters, then no `2*`
        memAdaptor.get(2 * loIx , loVec);
        memAdaptor.get(2 * hiIx, hiVec);
        complex_t<Scalar> lo = {loVec.x, loVec.y};  
        complex_t<Scalar> hi = {hiVec.x, hiVec.y};

        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps
        if (_NBL_HLSL_WORKGROUP_SIZE_ > glsl::gl_SubgroupSize())
        {
            // special first iteration
            hlsl::fft::DIF<Scalar>::radix2(hlsl::fft::twiddle<false, Scalar>(threadID, _NBL_HLSL_WORKGROUP_SIZE_), lo, hi);

            // Run bigger steps until Subgroup-sized
            for (uint32_t stride = _NBL_HLSL_WORKGROUP_SIZE_ >> 1; stride > glsl::gl_SubgroupSize(); stride >>= 1)
            {   
                FFT_loop<SharedMemoryAccessor>(stride, lo, hi, threadID, sharedmemAdaptor);
                sharedmemAdaptor.workgroupExecutionAndMemoryBarrier(); 
            }

            // special last workgroup-shuffle     
            exchangeValues<SharedMemoryAccessor, Scalar>(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAdaptor);  
        }

        // Subgroup-sized FFT
        subgroup::fft::FFT<false, Scalar, device_capabilities>::__call(lo, hi);

        // Put values back in global mem
        loVec = vector <Scalar, 2>(lo.real(), lo.imag());
        hiVec = vector <Scalar, 2>(hi.real(), hi.imag());

        memAdaptor.set(2 * loIx, loVec);
        memAdaptor.set(2 * hiIx, hiVec);

        // Update state for accessors
        accessor = memAdaptor.accessor;
        sharedmemAccessor = sharedmemAdaptor.accessor;
    }
};



// 2 items per invocation inverse specialization
template<typename Scalar, class device_capabilities>
struct FFT<2,true, Scalar, device_capabilities>
{
    template<typename SharedMemoryAccessor>
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, NBL_REF_ARG(MemoryAdaptor<SharedMemoryAccessor>) sharedmemAdaptor)
    {
        // Get twiddle with k = threadID mod stride, halfN = stride
        hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true, Scalar>(threadID & (stride - 1), stride), lo, hi);     
    
        exchangeValues<SharedMemoryAccessor, Scalar>(lo, hi, threadID, stride, sharedmemAdaptor);
    }


    template<typename Accessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        // Set up the MemAdaptors
        MemoryAdaptor<Accessor, 1> memAdaptor;
        memAdaptor.accessor = accessor;
        MemoryAdaptor<SharedMemoryAccessor> sharedmemAdaptor;
        sharedmemAdaptor.accessor = sharedmemAccessor;

        // Compute the indices only once
        const uint32_t threadID = uint32_t(SubgroupContiguousIndex());
        const uint32_t loIx = (glsl::gl_SubgroupID()<<(glsl::gl_SubgroupSizeLog2()+1))+glsl::gl_SubgroupInvocationID();
		const uint32_t hiIx = loIx+glsl::gl_SubgroupSize();

        // Read lo, hi values from global memory
        vector <Scalar, 2> loVec;
        vector <Scalar, 2> hiVec;
        memAdaptor.get(2 * loIx , loVec);
        memAdaptor.get(2 * hiIx, hiVec);
        complex_t<Scalar> lo = {loVec.x, loVec.y};  
        complex_t<Scalar> hi = {hiVec.x, hiVec.y}; 

        // Run a subgroup-sized FFT, then continue with bigger steps
        subgroup::fft::FFT<true, Scalar, device_capabilities>::__call(lo, hi);
        
        // If for some reason you're running a small FFT, skip all the bigger-than-subgroup steps

        if (_NBL_HLSL_WORKGROUP_SIZE_ > glsl::gl_SubgroupSize()) 
        { 
            // special first workgroup-shuffle
            exchangeValues<SharedMemoryAccessor, Scalar>(lo, hi, threadID, glsl::gl_SubgroupSize(), sharedmemAdaptor);
        
            // The bigger steps
            for (uint32_t stride = glsl::gl_SubgroupSize() << 1; stride < _NBL_HLSL_WORKGROUP_SIZE_; stride <<= 1)
            {   
                // Order of waiting for shared mem writes is also reversed here, since the shuffle came earlier
                sharedmemAdaptor.workgroupExecutionAndMemoryBarrier(); 
                FFT_loop<SharedMemoryAccessor>(stride, lo, hi, threadID, sharedmemAdaptor);
            }

            // special last iteration 
            hlsl::fft::DIT<Scalar>::radix2(hlsl::fft::twiddle<true, Scalar>(threadID, _NBL_HLSL_WORKGROUP_SIZE_), lo, hi); 
            divides_assign< complex_t<Scalar> > divAss;
            divAss(lo, _NBL_HLSL_WORKGROUP_SIZE_ / glsl::gl_SubgroupSize());
            divAss(hi, _NBL_HLSL_WORKGROUP_SIZE_ / glsl::gl_SubgroupSize());      
        }   
        
        // Put values back in global mem
        loVec = vector <Scalar, 2>(lo.real(), lo.imag());
        hiVec = vector <Scalar, 2>(hi.real(), hi.imag());
        memAdaptor.set(2 * loIx, loVec);
        memAdaptor.set(2 * hiIx, hiVec);

        // Update state for accessors
        accessor = memAdaptor.accessor;
        sharedmemAccessor = sharedmemAdaptor.accessor;
    }
};


// ---------------------------- Below pending --------------------------------------------------

/*

// then define 4,8,16 in terms of calling the FFT<2> and doing the special radix steps before/after
template<uint16_t K, bool Inverse, class device_capabilities>
struct FFT
{
    template<typename Accessor, typename ShaderMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(ShaderMemoryAccessor) sharedmemAccessor)
    {
        if (!Inverse)
        {
           ... special steps ...
        }
        FFT<2,Inverse,device_capabilities>::template __call<Accessor,SharedMemoryAccessor>(access,sharedMemAccessor);
        if (Inverse)
        {
           ... special steps ...
        }
    }
};

*/

}
}
}
}

#endif