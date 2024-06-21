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

/*

template<typename Accessor, typename Scalar, uint32_t WorkgroupSize>
void store(NBL_REF_ARG(Accessor) sharedmemAccessor, uint32_t threadID, NBL_CONST_REF_ARG(complex_t<Scalar>) lo, NBL_CONST_REF_ARG(complex_t<Scalar>) hi) {
    sharedmemAccessor.set(threadID + 0 * WorkgroupSize, lo.real());
    sharedmemAccessor.set(threadID + 1 * WorkgroupSize, hi.real());
    sharedmemAccessor.set(threadID + 2 * WorkgroupSize, lo.imag());  
    sharedmemAccessor.set(threadID + 3 * WorkgroupSize, hi.imag());    
}

template<typename Accessor, typename Scalar, uint32_t WorkgroupSize>
void load(NBL_CONST_REF_ARG(Accessor) sharedmemAccessor, uint32_t threadID, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi) {
    lo.real(sharedmemAccessor.get(threadID + 0 * WorkgroupSize));
    hi.real(sharedmemAccessor.get(threadID + 1 * WorkgroupSize));
    lo.imag(sharedmemAccessor.get(threadID + 2 * WorkgroupSize));  
    hi.imag(sharedmemAccessor.get(threadID + 3 * WorkgroupSize));    
}

template<typename Accessor, typename Scalar, uint32_t WorkgroupSize, bool inverse>
void FFT_loop(NBL_REF_ARG(Accessor) sharedmemAccessor, uint32_t step, uint32_t threadID){
    const uint32_t sub_ix = threadID & (step - 1);
    const uint32_t lo_x_ix = impl::bitfieldInsert(threadID, 0u ,sub_ix, 1u);
    const uint32_t hi_x_ix = lo_x_ix | step;
    const uint32_t lo_y_ix = lo_x_ix + 2 * WorkgroupSize;
    const uint32_t hi_y_ix = hi_x_ix + 2 * WorkgroupSize;

    sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
    complex_t<Scalar> lo = {sharedmemAccessor.get(lo_x_ix), sharedmemAccessor.get(lo_y_ix)};
    complex_t<Scalar> hi = {sharedmemAccessor.get(hi_x_ix), sharedmemAccessor.get(hi_y_ix)};

    fft::DIX<Scalar, inverse>::radix2(fft::twiddle<Scalar, inverse>(sub_ix, step << 1), lo, hi);

    // Share results between threads
    sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
    sharedmemAccessor.set(lo_x_ix, lo.real());
    sharedmemAccessor.set(hi_x_ix, hi.real());
    sharedmemAccessor.set(lo_y_ix, lo.imag());
    sharedmemAccessor.set(hi_y_ix, hi.imag());     
}

// When doing a 1D FFT of size N, we call N/2 threads to do the work, where each thread of index 0 <= k < N/2 is in charge of computing two elements
// in the output array, those of index k and k + N/2. The first is "lo" and the latter is "hi"
template<typename Accessor, typename Scalar, uint32_t WorkgroupSize, uint32_t SubgroupSize, bool inverse>
void FFT(NBL_REF_ARG(Accessor) sharedmemAccessor, NBL_REF_ARG(complex_t<Scalar>) lo, NBL_REF_ARG(complex_t<Scalar>) hi){
    const uint32_t doubleWorkgroupSize = WorkgroupSize << 1;
    const uint32_t threadID = glsl::gl_LocalInvocationIndex();

    if (inverse)
    {
        // Run a subgroup-sized FFT, then continue with bigger steps
        subgroup::FFT<Scalar, SubgroupSize, true>(lo, hi);
        
        // Put values into workgroup shared smem
        sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
        store<Accessor, Scalar, WorkgroupSize>(sharedmemAccessor, threadID, lo, hi);             

        // The bigger steps
        for (uint32_t step = SubgroupSize << 1; step < WorkgroupSize; step <<= 1)
            FFT_loop<Accessor, Scalar, WorkgroupSize, true>(sharedmemAccessor, step, threadID);

        // Get values back from smem
        sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
        load<Accessor, Scalar, WorkgroupSize>(sharedmemAccessor, threadID, lo, hi); 
        sharedmemAccessor.workgroupExecutionAndMemoryBarrier();

        // special last iteration
        fft::DIT<Scalar>::radix2(fft::twiddle<Scalar, true>(threadID, doubleWorkgroupSize), lo, hi); 
        divides_assign< complex_t<Scalar> > divAss;
        divAss(lo, WorkgroupSize / SubgroupSize);
        divAss(hi, WorkgroupSize / SubgroupSize);
    }
    else
    {
        // special first iteration
        fft::DIF<Scalar>::radix2(fft::twiddle<Scalar, false>(threadID, doubleWorkgroupSize), lo, hi); 

        // Put values into workgroup shared smem
        sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
        store<Accessor, Scalar, WorkgroupSize>(sharedmemAccessor, threadID, lo, hi);
        
        // Run bigger steps until Subgroup-sized
        for (uint32_t step = WorkgroupSize >> 1; step > SubgroupSize; step >>= 1)
            FFT_loop<Accessor, Scalar, WorkgroupSize, false>(sharedmemAccessor, step, threadID);

        // Get values back from smem
        sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
        load<Accessor, Scalar, WorkgroupSize>(sharedmemAccessor, threadID, lo, hi); 
        sharedmemAccessor.workgroupExecutionAndMemoryBarrier();

        // Subgroup-sized FFT
        subgroup::FFT<Scalar, SubgroupSize, false>(lo, hi);
    }
}

*/

// ----------------------------------------- ABOVE MARKED FOR DELETION -----------------------------------------------------

template<uint16_t ElementsPerInvocation, bool Inverse, typneame Scalar, class device_capabilities=void>
struct FFT;

// For the FFT methods below, we assume:
//      - Accessor is a global memory accessor to an array fitting 2 * _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar>, used to get inputs / set outputs of the FFT
//      - SharedMemoryAccessor accesses a shared memory array that can fit _NBL_HLSL_WORKGROUP_SIZE_ elements of type complex_t<Scalar> 

// 2 items per invocation forward specialization
template<typename Scalar, class device_capabilities>
struct FFT<2,false,device_capabilities>
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
        subgroup::FFT<2, false, Scalar, device_capabilities>::__call(lo, hi);

        // Put values back in global mem
        loHiPacked = vector <Scalar, 4>(lo.real(), lo.imag(), hi.real(), hi.imag());
        memAdaptor.set(threadID, loHiPacked);

        // Update state for accessors
        accessor = memAdaptor.accessor;
        sharedmemAccessor = sharedmemAdaptor.accessor;
    }
};



// 2 items per invocation inverse specialization
template<class device_capabilities>
struct FFT<2,true,device_capabilities>
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
        subgroup::FFT<2, true, Scalar, device_capabilities>::__call(lo, hi);
        
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