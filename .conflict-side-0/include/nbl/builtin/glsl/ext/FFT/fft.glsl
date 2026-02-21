// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_INCLUDED_
#define _NBL_GLSL_EXT_FFT_INCLUDED_

#include <nbl/builtin/glsl/math/complex.glsl>
#include <nbl/builtin/glsl/ext/FFT/parameters.glsl>

#ifndef _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_
#error "_NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ should be defined."
#endif

#include "nbl/builtin/glsl/workgroup/shared_fft.glsl"


#ifndef _NBL_GLSL_EXT_FFT_SET_DATA_DECLARED_
#define _NBL_GLSL_EXT_FFT_SET_DATA_DECLARED_
void nbl_glsl_ext_FFT_setData(in uvec3 coordinate, in uint channel, in vec2 complex_value);
#endif

#ifndef _NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DECLARED_
#define _NBL_GLSL_EXT_FFT_GET_PADDED_DATA_DECLARED_
nbl_glsl_complex nbl_glsl_ext_FFT_getPaddedData(in ivec3 coordinate, in uint channel);
#endif

#ifndef _NBL_GLSL_EXT_FFT_GET_PARAMETERS_DEFINED_
#error "You need to define `nbl_glsl_ext_FFT_getParameters` and mark `_NBL_GLSL_EXT_FFT_GET_PARAMETERS_DEFINED_`!"
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
ivec3 nbl_glsl_ext_FFT_getPaddedCoordinates(in uint tidx, in uint log2FFTSize, in uint trueDimension)
{
    const uint padding = ((0x1u<<log2FFTSize)-trueDimension)>>1u;
    return ivec3(nbl_glsl_ext_FFT_getCoordinates(tidx-padding));
}


#include "nbl/builtin/glsl/workgroup/fft.glsl"


nbl_glsl_complex nbl_glsl_ext_FFT_impl_values[(_NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_-1u)/_NBL_GLSL_WORKGROUP_SIZE_+1u]; // TODO: hardcoded channel count, multichannel FFT (shared twiddles), maybe even precompute twiddles into a LUT!?

void nbl_glsl_ext_FFT_loop(in bool is_inverse, in uint virtual_thread_count, in uint step)
{
    for(uint t=0u; t<virtual_thread_count; t++)
    {
        const uint pseudo_step = step>>_NBL_GLSL_WORKGROUP_SIZE_LOG2_;
        const uint lo = t&(pseudo_step-1u);
        const uint lo_ix = nbl_glsl_bitfieldInsert_impl(t,0u,lo,1u);
        const uint hi_ix = lo_ix|pseudo_step;
        
        const uint subFFTItem = (lo<<_NBL_GLSL_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
        nbl_glsl_complex twiddle = nbl_glsl_FFT_twiddle(is_inverse,subFFTItem,float(step<<1u));
        if (is_inverse)
            nbl_glsl_FFT_DIT_radix2(twiddle,nbl_glsl_ext_FFT_impl_values[lo_ix],nbl_glsl_ext_FFT_impl_values[hi_ix]);
        else
            nbl_glsl_FFT_DIF_radix2(twiddle,nbl_glsl_ext_FFT_impl_values[lo_ix],nbl_glsl_ext_FFT_impl_values[hi_ix]);
    }
}

void nbl_glsl_ext_FFT_preloaded(bool is_inverse, in uint log2FFTSize)
{
    // Virtual Threads Calculation
    const uint dataLength = 1u<<log2FFTSize;
    const uint halfDataLength = dataLength>>1u;
    const uint virtual_thread_count = halfDataLength>>_NBL_GLSL_WORKGROUP_SIZE_LOG2_;

    // special forward steps
    if (!is_inverse)
    for (uint step=halfDataLength; step>_NBL_GLSL_WORKGROUP_SIZE_; step>>=1u)
        nbl_glsl_ext_FFT_loop(false,virtual_thread_count,step);
    // do workgroup sized sub-FFTs
    for(uint t=0u; t<virtual_thread_count; t++)
    {
        const uint lo_ix = t<<1u;
        const uint hi_ix = lo_ix|1u;
        nbl_glsl_workgroupFFT(is_inverse,nbl_glsl_ext_FFT_impl_values[lo_ix],nbl_glsl_ext_FFT_impl_values[hi_ix]);
        if (is_inverse)
        {
            nbl_glsl_ext_FFT_impl_values[lo_ix] /= float(virtual_thread_count);
            nbl_glsl_ext_FFT_impl_values[hi_ix] /= float(virtual_thread_count);
        }
    }
    // special inverse steps
    if (is_inverse)
    for (uint step=_NBL_GLSL_WORKGROUP_SIZE_<<1u; step<dataLength; step<<=1u)
        nbl_glsl_ext_FFT_loop(true,virtual_thread_count,step);
}

void nbl_glsl_ext_FFT(bool is_inverse, uint channel)
{
    // Virtual Threads Calculation
    const uint log2FFTSize = nbl_glsl_ext_FFT_Parameters_t_getLog2FFTSize();
    const uint item_per_thread_count = 0x1u<<(log2FFTSize-_NBL_GLSL_WORKGROUP_SIZE_LOG2_);

    // Load Values into local memory
    for(uint t=0u; t<item_per_thread_count; t++)
    {
        const uint tid = (t<<_NBL_GLSL_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
        const uint trueDim = nbl_glsl_ext_FFT_Parameters_t_getDimensions()[nbl_glsl_ext_FFT_Parameters_t_getDirection()];
        nbl_glsl_ext_FFT_impl_values[t] = nbl_glsl_ext_FFT_getPaddedData(nbl_glsl_ext_FFT_getPaddedCoordinates(tid,log2FFTSize,trueDim),channel);
    }
    // do FFT
    nbl_glsl_ext_FFT_preloaded(is_inverse,log2FFTSize);
    // write out to main memory
    for(uint t=0u; t<item_per_thread_count; t++)
    {
        const uint tid = (t<<_NBL_GLSL_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
        nbl_glsl_ext_FFT_setData(nbl_glsl_ext_FFT_getCoordinates(tid),channel,nbl_glsl_ext_FFT_impl_values[t]);
    }
}

#endif