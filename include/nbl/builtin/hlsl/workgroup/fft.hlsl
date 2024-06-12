#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_FFT_INCLUDED_

#include <nbl/builtin/hlsl/subgroup/fft.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

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
        //divAss(lo, doubleWorkgroupSize);
        //divAss(hi, doubleWorkgroupSize);
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


}
}
}

#endif