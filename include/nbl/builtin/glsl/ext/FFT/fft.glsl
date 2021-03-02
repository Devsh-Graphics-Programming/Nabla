// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_INCLUDED_
#define _NBL_GLSL_EXT_FFT_INCLUDED_

#include <nbl/builtin/glsl/math/complex.glsl>

#include <nbl/builtin/glsl/macros.glsl>
#include <nbl/builtin/glsl/ext/FFT/parameters.glsl>

#ifndef _NBL_GLSL_EXT_FFT_MAX_CHANNELS
#error "_NBL_GLSL_EXT_FFT_MAX_CHANNELS should be defined."
#endif

#ifndef _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_
#error "_NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ should be defined."
#endif

#ifndef _NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD
#error "_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD should be defined."
#endif

#ifndef _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_
#error "_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_ should be defined."
#endif

// TODO: can we reduce it?
#define _NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_ (_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*4)

#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
    #if NBL_GLSL_LESS(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_)
        #error "Not enough shared memory declared for ext::FFT !"
    #endif
#else
    #if NBL_GLSL_GREATER(_NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_,0)
        #define _NBL_GLSL_SCRATCH_SHARED_DEFINED_ nbl_glsl_fftScratchShared
        #define _NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_ _NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_
        shared uint _NBL_GLSL_SCRATCH_SHARED_DEFINED_[_NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_];
    #endif
#endif

// Push Constants

#define _NBL_GLSL_EXT_FFT_DIRECTION_X_ 0
#define _NBL_GLSL_EXT_FFT_DIRECTION_Y_ 1
#define _NBL_GLSL_EXT_FFT_DIRECTION_Z_ 2

#define _NBL_GLSL_EXT_FFT_CLAMP_TO_EDGE_ 0
#define _NBL_GLSL_EXT_FFT_FILL_WITH_ZERO_ 1


#ifndef _NBL_GLSL_EXT_FFT_GET_DATA_DECLARED_
#define _NBL_GLSL_EXT_FFT_GET_DATA_DECLARED_
vec2 nbl_glsl_ext_FFT_getData(in uvec3 coordinate, in uint channel);
#endif

#ifndef _NBL_GLSL_EXT_FFT_SET_DATA_DECLARED_
#define _NBL_GLSL_EXT_FFT_SET_DATA_DECLARED_
void nbl_glsl_ext_FFT_setData(in uvec3 coordinate, in uint channel, in vec2 complex_value);
#endif

#ifndef _NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DECLARED_
#define _NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DECLARED_
vec2 nbl_glsl_ext_FFT_getPaddedData(in uvec3 coordinate, in uint channel);
#endif

#ifndef _NBL_GLSL_EXT_FFT_GET_PARAMETERS_DEFINED_
#error "You need to define `nbl_glsl_ext_FFT_getParameters` and mark `_NBL_GLSL_EXT_FFT_GET_PARAMETERS_DEFINED_`!"
#endif
#ifndef _NBL_GLSL_EXT_FFT_GET_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_FFT_getData` and mark `_NBL_GLSL_EXT_FFT_GET_DATA_DEFINED_`!"
#endif
#ifndef _NBL_GLSL_EXT_FFT_SET_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_FFT_setData` and mark `_NBL_GLSL_EXT_FFT_SET_DATA_DEFINED_`!"
#endif
#ifndef _NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_FFT_getPaddedData` and mark `_NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DEFINED_`!"
#endif

uvec3 nbl_glsl_ext_FFT_getCoordinates(in uint tidx)
{
    uint direction = nbl_glsl_ext_FFT_Parameters_t_getDirection();
    uvec3 tmp = gl_WorkGroupID;
    tmp[direction] = tidx;
    return tmp;
}

uint nbl_glsl_ext_FFT_calculateTwiddlePower(in uint threadId, in uint iteration, in uint logTwoN) 
{
    const uint shiftSuffix = logTwoN - 1u - iteration;
    const uint suffixMask = (1u << iteration) - 1u;
    return (threadId & suffixMask) << shiftSuffix;
}

uint nbl_glsl_ext_FFT_getEvenIndex(in uint threadId, in uint iteration, in uint N) {
    return ((threadId & (N - (1u << iteration))) << 1u) | (threadId & ((1u << iteration) - 1u));
}

// TODO: temp
#define _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_ findMSB(_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_)
//TODO: optimization for DFT of real signal


void nbl_glsl_workgroup_FFT_loop(in bool is_inverse, in uint stride)
{
    const uint sub_ix = gl_LocalInvocationIndex&(stride-1u);
    const uint lo_x_ix = nbl_glsl_bitfieldInsert_impl(gl_LocalInvocationIndex,0u,sub_ix,1u);
    const uint hi_x_ix = lo_x_ix|stride;
    const uint lo_y_ix = lo_x_ix+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2;
    const uint hi_y_ix = hi_x_ix+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2;
            
    nbl_glsl_complex low = nbl_glsl_complex(uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_x_ix]),uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_y_ix]));
    nbl_glsl_complex high = nbl_glsl_complex(uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_x_ix]),uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_y_ix]));

    nbl_glsl_complex twiddle = nbl_glsl_FFT_twiddle(is_inverse,sub_ix,float(stride<<1u));
    if (is_inverse)
        nbl_glsl_FFT_DIT_radix2(twiddle,low,high);
    else
        nbl_glsl_FFT_DIF_radix2(twiddle,low,high);

    barrier();
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_x_ix] = floatBitsToUint(low.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_x_ix] = floatBitsToUint(high.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_y_ix] = floatBitsToUint(low.y);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_y_ix] = floatBitsToUint(high.y);
    barrier();
}
// Decimates in Frequency for forward transform, in Time for reverse, hence no bitreverse permutation needed
void nbl_glsl_workgroup_FFT(in bool is_inverse, inout nbl_glsl_complex lo, inout nbl_glsl_complex hi)
{
    const float doubleWorkgroupSize = float(_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_<<1u);
    // special first iteration
    if (!is_inverse)
        nbl_glsl_FFT_DIF_radix2(nbl_glsl_FFT_twiddle(false,gl_LocalInvocationIndex,doubleWorkgroupSize),lo,hi);

    const uint tid = gl_LocalInvocationIndex;
    // get the values in
    barrier();
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*0u] = floatBitsToUint(lo.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*1u] = floatBitsToUint(hi.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2u] = floatBitsToUint(lo.y);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*3u] = floatBitsToUint(hi.y);
    barrier();

    // TODO: use subgroup_FFT op to reduce bank conflicts in the final iterations
    if (is_inverse)
    {
        for (uint step=1u; step<_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_; step<<=1u)
            nbl_glsl_workgroup_FFT_loop(true,step);
    }
    else
    {
        for (uint step=_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_>>1u; step>0u; step>>=1u)
            nbl_glsl_workgroup_FFT_loop(false,step);
    }

    // get the values out
    lo = nbl_glsl_complex(
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*0u]),
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2u])
    );
    hi = nbl_glsl_complex(
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*1u]),
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*3u])
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


nbl_glsl_complex nbl_glsl_ext_FFT_impl_values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD*2u]; // TODO: redo later
void nbl_glsl_ext_FFT_loop(in bool is_inverse, in uint virtual_thread_count, in uint step)
{
    for(uint t=0u; t<virtual_thread_count; t++)
    {
        const uint pseudo_step = step>>_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_;
        const uint lo = t&(pseudo_step-1u);
        const uint lo_ix = nbl_glsl_bitfieldInsert_impl(t,0u,lo,1u);
        const uint hi_ix = lo_ix|pseudo_step;
        
        const uint subFFTItem = (lo<<_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
        nbl_glsl_complex twiddle = nbl_glsl_FFT_twiddle(is_inverse,subFFTItem,float(step<<1u));
        if (is_inverse)
            nbl_glsl_FFT_DIT_radix2(twiddle,nbl_glsl_ext_FFT_impl_values[lo_ix],nbl_glsl_ext_FFT_impl_values[hi_ix]);
        else
            nbl_glsl_FFT_DIF_radix2(twiddle,nbl_glsl_ext_FFT_impl_values[lo_ix],nbl_glsl_ext_FFT_impl_values[hi_ix]);
    }
}
// TODO: try radix-4 or even radix-8 for perf
void nbl_glsl_ext_FFT(bool is_inverse, uint channel)
{
    // Virtual Threads Calculation
    const uint dataLength = nbl_glsl_ext_FFT_Parameters_t_getFFTLength();
    const uint item_per_thread_count = dataLength>>_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_;

    const uint halfDataLength = dataLength>>1u;
    const uint virtual_thread_count = item_per_thread_count>>1u;

    // Load Values into local memory
    for(uint t=0u; t<item_per_thread_count; t++)
    {
        const uint tid = (t<<_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
        nbl_glsl_ext_FFT_impl_values[t] = nbl_glsl_ext_FFT_getPaddedData(nbl_glsl_ext_FFT_getCoordinates(tid),channel);
        if (is_inverse)
            nbl_glsl_ext_FFT_impl_values[t] /= float(virtual_thread_count);
    }
    // special forward steps
    if (!is_inverse)
    for (uint step = halfDataLength; step > _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_; step >>= 1u)
        nbl_glsl_ext_FFT_loop(false,virtual_thread_count,step);
    // do workgroup sized sub-FFTs
    for(uint t=0u; t<virtual_thread_count; t++)
    {
        const uint lo_ix = t<<1u;
        const uint hi_ix = lo_ix|1u;
        nbl_glsl_workgroup_FFT(is_inverse,nbl_glsl_ext_FFT_impl_values[lo_ix],nbl_glsl_ext_FFT_impl_values[hi_ix]);
    }
    // special inverse steps
    if (is_inverse)
    for (uint step=_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_<<1u; step<dataLength; step<<=1u)
        nbl_glsl_ext_FFT_loop(true,virtual_thread_count,step);
    // write out to main memory
    for(uint t=0u; t<item_per_thread_count; t++)
    {
        const uint tid = (t<<_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
        nbl_glsl_ext_FFT_setData(nbl_glsl_ext_FFT_getCoordinates(tid),channel,nbl_glsl_ext_FFT_impl_values[t]);
    }
}

#endif