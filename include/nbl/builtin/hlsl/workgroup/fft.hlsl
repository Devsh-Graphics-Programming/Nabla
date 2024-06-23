#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

template<uint16_t ElementsPerInvocation, bool Inverse, typename Scalar, class device_capabilities=void>
struct FFT;

// For the FFT methods below, we assume:
//      - Accessor is a global memory accessor to an array fitting 2 * _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar>, used to get inputs / set outputs of the FFT
//      - SharedMemoryAccessor accesses a shared memory array that can fit _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar> 

// 2 items per invocation forward specialization
template<typename Scalar, class device_capabilities>
struct FFT<2,false, Scalar, device_capabilities>
{
    template<typename SharedMemoryAccessor>
    static void FFT_loop(uint32_t stride, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi, uint32_t threadID, NBL_REF_ARG(MemoryAdaptor<SharedMemoryAccessor>) sharedmemAdaptor)
    {
        const bool topHalf = bool(threadID & stride);
        vector <Scalar, 2> toTrade = topHalf ? vector <Scalar, 2>(lo.real(), lo.imag()) : vector <Scalar, 2>(hi.real(), hi.imag());
        
        // Block memory writes until all threads are done with previous work
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();
        sharedmemAdaptor.set(threadID, toTrade);
        
        // Wait until all writes are done before reading
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();
        sharedmemAdaptor.get(threadID ^ stride, toTrade);
        
        if (topHalf)
        {
            lo.real(toTrade.x);
            lo.imag(toTrade.y);
        }
        else
        {
            hi.real(toTrade.x);
            hi.imag(toTrade.y);
        }
        // Get twiddle with k = threadID mod stride, halfN = stride
        fft::DIF<Scalar>::radix2(fft::twiddle<false, Scalar>(threadID & (stride - 1), stride), lo, hi);    
    }


    template<typename Accessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        // Set up the MemAdaptors
        MemoryAdaptor<Accessor> memAdaptor;
        memAdaptor.accessor = accessor;
        MemoryAdaptor<SharedMemoryAccessor> sharedmemAdaptor;
        sharedmemAdaptor.accessor = sharedmemAccessor;

        // Compute the SubgroupContiguousIndex only once
        const uint32_t threadID = uint32_t(SubgroupContiguousIndex());

        // Read lo, hi values from global memory
        vector <Scalar, 4> loHiPacked;
        memAdaptor.get(threadID, loHiPacked);
        complex_t<Scalar> lo = {loHiPacked.x, loHiPacked.y};  
        complex_t<Scalar> hi = {loHiPacked.z, loHiPacked.w};

        // special first iteration
        if (_NBL_HLSL_WORKGROUP_SIZE_ > glsl::gl_SubgroupSize())
            fft::DIF<Scalar>::radix2(fft::twiddle<false, Scalar>(threadID, _NBL_HLSL_WORKGROUP_SIZE_), lo, hi); 

        // Run bigger steps until Subgroup-sized
        for (uint32_t stride = _NBL_HLSL_WORKGROUP_SIZE_ >> 1; stride > glsl::gl_SubgroupSize(); stride >>= 1)
            FFT_loop<SharedMemoryAccessor>(stride, lo, hi, threadID, sharedmemAdaptor);
  
        // Subgroup-sized FFT
        subgroup::FFT<false, Scalar, device_capabilities>::__call(lo, hi);

        // Put values back in global mem
        loHiPacked = vector <Scalar, 4>(lo.real(), lo.imag(), hi.real(), hi.imag());
        memAdaptor.set(threadID, loHiPacked);

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
        fft::DIF<Scalar>::radix2(fft::twiddle<true, Scalar>(threadID & (stride - 1), stride), lo, hi);     
    
        const bool topHalf = bool(threadID & stride);
        vector <Scalar, 2> toTrade = topHalf ? vector <Scalar, 2>(lo.real(), lo.imag()) : vector <Scalar, 2>(hi.real(), hi.imag());
        
        // Block memory writes until all threads are sync'd
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();
        sharedmemAdaptor.set(threadID, toTrade);
        
        // Wait until all writes are done before reading
        sharedmemAdaptor.workgroupExecutionAndMemoryBarrier();
        sharedmemAdaptor.get(threadID ^ stride, toTrade);
        
        if (topHalf)
        {
            lo.real(toTrade.x);
            lo.imag(toTrade.y);
        }
        else
        {
            hi.real(toTrade.x);
            hi.imag(toTrade.y);
        }  
    }


    template<typename Accessor, typename SharedMemoryAccessor>
    static void __call(NBL_REF_ARG(Accessor) accessor, NBL_REF_ARG(SharedMemoryAccessor) sharedmemAccessor)
    {
        // Set up the MemAdaptors
        MemoryAdaptor<Accessor> memAdaptor;
        memAdaptor.accessor = accessor;
        MemoryAdaptor<SharedMemoryAccessor> sharedmemAdaptor;
        sharedmemAdaptor.accessor = sharedmemAccessor;

        // Compute the SubgroupContiguousIndex only once
        const uint32_t threadID = uint32_t(SubgroupContiguousIndex());

        // Read lo, hi values from global memory
        vector <Scalar, 4> loHiPacked;
        memAdaptor.get(threadID, loHiPacked);
        complex_t<Scalar> lo = {loHiPacked.x, loHiPacked.y};  
        complex_t<Scalar> hi = {loHiPacked.z, loHiPacked.w}; 

        // Run a subgroup-sized FFT, then continue with bigger steps
        subgroup::FFT<true, Scalar, device_capabilities>::__call(lo, hi);
        
        // The bigger steps
        for (uint32_t stride = glsl::gl_SubgroupSize() << 1; stride < _NBL_HLSL_WORKGROUP_SIZE_; stride <<= 1)
            FFT_loop<SharedMemoryAccessor>(stride, lo, hi, threadID, sharedmemAdaptor);

        // special last iteration
        if (_NBL_HLSL_WORKGROUP_SIZE_ > glsl::gl_SubgroupSize())
        {
            fft::DIT<Scalar>::radix2(fft::twiddle<Scalar, true>(threadID, _NBL_HLSL_WORKGROUP_SIZE_), lo, hi); 
            divides_assign< complex_t<Scalar> > divAss;
            divAss(lo, _NBL_HLSL_WORKGROUP_SIZE_ / glsl::gl_SubgroupSize());
            divAss(hi, _NBL_HLSL_WORKGROUP_SIZE_ / glsl::gl_SubgroupSize());
        }
        
        // Put values back in global mem
        loHiPacked = vector <Scalar, 4>(lo.real(), lo.imag(), hi.real(), hi.imag());
        memAdaptor.set(threadID, loHiPacked);

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

#endif