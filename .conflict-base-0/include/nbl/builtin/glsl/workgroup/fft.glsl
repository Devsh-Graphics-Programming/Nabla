// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_GLSL_WORKGROUP_FFT_INCLUDED_
#define _NBL_BUILTIN_GLSL_WORKGROUP_FFT_INCLUDED_



#include <nbl/builtin/glsl/workgroup/shared_fft.glsl>


#include <nbl/builtin/glsl/math/complex.glsl>
#include <nbl/builtin/glsl/subgroup/basic_portability.glsl>
//#include <nbl/builtin/glsl/subgroup/fft.glsl>


#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
    #if NBL_GLSL_LESS(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_NBL_GLSL_WORKGROUP_FFT_SHARED_SIZE_NEEDED_)
        #error "Not enough shared memory declared for workgroup FFT!"
    #endif
#else
    #if NBL_GLSL_GREATER(_NBL_GLSL_WORKGROUP_FFT_SHARED_SIZE_NEEDED_,0)
        #define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_workgroupFFTScratchShared
        #define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_WORKGROUP_FFT_SHARED_SIZE_NEEDED_
        shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_];
    #endif
#endif


//TODO: try radix-4 or even radix-8 for perf

void nbl_glsl_workgroupFFT_loop(in bool is_inverse, in uint stride)
{
    barrier();
    const uint sub_ix = gl_LocalInvocationIndex&(stride-1u);
    const uint lo_x_ix = nbl_glsl_bitfieldInsert_impl(gl_LocalInvocationIndex,0u,sub_ix,1u);
    const uint hi_x_ix = lo_x_ix|stride;
    const uint lo_y_ix = lo_x_ix+_NBL_GLSL_WORKGROUP_SIZE_*2;
    const uint hi_y_ix = hi_x_ix+_NBL_GLSL_WORKGROUP_SIZE_*2;
            
    nbl_glsl_complex low = nbl_glsl_complex(uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_x_ix]),uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_y_ix]));
    nbl_glsl_complex high = nbl_glsl_complex(uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_x_ix]),uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_y_ix]));

    nbl_glsl_complex twiddle = nbl_glsl_complex(1.f,0.f);
    if (stride!=1u)
        twiddle = nbl_glsl_FFT_twiddle(is_inverse,sub_ix,float(stride<<1u));

    if (is_inverse)
        nbl_glsl_FFT_DIT_radix2(twiddle,low,high);
    else
        nbl_glsl_FFT_DIF_radix2(twiddle,low,high);

    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_x_ix] = floatBitsToUint(low.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_x_ix] = floatBitsToUint(high.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_y_ix] = floatBitsToUint(low.y);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_y_ix] = floatBitsToUint(high.y);
}
//! Decimates in Frequency for forward transform, in Time for reverse, hence no bitreverse permutation needed
void nbl_glsl_workgroupFFT(in bool is_inverse, inout nbl_glsl_complex lo, inout nbl_glsl_complex hi)
{
    const float doubleWorkgroupSize = float(_NBL_GLSL_WORKGROUP_SIZE_<<1u);
    // special first iteration
    if (!is_inverse)
        nbl_glsl_FFT_DIF_radix2(nbl_glsl_FFT_twiddle(false,gl_LocalInvocationIndex,doubleWorkgroupSize),lo,hi);

    const uint tid = gl_LocalInvocationIndex;
    // get the values in
    barrier();
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_WORKGROUP_SIZE_*0u] = floatBitsToUint(lo.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_WORKGROUP_SIZE_*1u] = floatBitsToUint(hi.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_WORKGROUP_SIZE_*2u] = floatBitsToUint(lo.y);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_WORKGROUP_SIZE_*3u] = floatBitsToUint(hi.y);

    // TODO: use subgroup_FFT op to reduce bank conflicts in the final iterations
    if (is_inverse)
    {
        for (uint step=1u; step<_NBL_GLSL_WORKGROUP_SIZE_; step<<=1u)
            nbl_glsl_workgroupFFT_loop(true,step);
    }
    else
    {
        for (uint step=_NBL_GLSL_WORKGROUP_SIZE_>>1u; step>0u; step>>=1u)
            nbl_glsl_workgroupFFT_loop(false,step);
    }

    // get the values out
    barrier();
    lo = nbl_glsl_complex(
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_WORKGROUP_SIZE_*0u]),
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_WORKGROUP_SIZE_*2u])
    );
    hi = nbl_glsl_complex(
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_WORKGROUP_SIZE_*1u]),
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_WORKGROUP_SIZE_*3u])
    );
    barrier();
        
    // special last iteration
    if (is_inverse)
    {
        nbl_glsl_FFT_DIT_radix2(nbl_glsl_FFT_twiddle(true,gl_LocalInvocationIndex,doubleWorkgroupSize),lo,hi);

        lo /= doubleWorkgroupSize;
        hi /= doubleWorkgroupSize;
    }
}


#endif