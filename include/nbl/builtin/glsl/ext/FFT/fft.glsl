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

#define _NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_ _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_

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
void nbl_glsl_workgroup_FFT_DIF(bool is_inverse, inout nbl_glsl_complex lo, inout nbl_glsl_complex hi)
{
    // special first iteration
    nbl_glsl_FFT_DIF_radix2(nbl_glsl_FFT_twiddle(is_inverse,gl_LocalInvocationIndex,uint(_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_<<1u)),lo,hi);
    
    barrier();
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*0u] = floatBitsToUint(lo.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*1u] = floatBitsToUint(hi.x);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2u] = floatBitsToUint(lo.y);
    _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*3u] = floatBitsToUint(hi.y);
    barrier();

    // TODO: use subgroup_FFT op to reduce bank conflicts in the final iterations
    for (uint step=_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_>>1u; step>0u; step>>=1u)
    {
        const uint sub_ix = gl_LocalInvocationIndex&(step-1u);
        const uint lo_w_ix = nbl_glsl_bitfieldInsert_impl(gl_LocalInvocationIndex,0u,sub_ix,1u);
        const uint hi_w_ix = lo_w_ix|step;
            
        nbl_glsl_complex low = nbl_glsl_complex(uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_w_ix]),uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_w_ix+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2]));
        nbl_glsl_complex high = nbl_glsl_complex(uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_w_ix]),uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_w_ix+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2]));
        nbl_glsl_FFT_DIF_radix2(nbl_glsl_FFT_twiddle(is_inverse,sub_ix,float(step<<1u)),low,high);
        barrier();
        /* this makes stockham or something
        _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*0u] = floatBitsToUint(low.x);
        _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*1u] = floatBitsToUint(high.x);
        _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2u] = floatBitsToUint(low.y);
        _NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*3u] = floatBitsToUint(high.y);
        */
        _NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_w_ix] = floatBitsToUint(low.x);
        _NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_w_ix] = floatBitsToUint(high.x);
        _NBL_GLSL_SCRATCH_SHARED_DEFINED_[lo_w_ix+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2] = floatBitsToUint(low.y);
        _NBL_GLSL_SCRATCH_SHARED_DEFINED_[hi_w_ix+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2] = floatBitsToUint(high.y);
        barrier();
    }
    /* stockham
    const uint tid = gl_LocalInvocationIndex;
    lo = nbl_glsl_complex(
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*0u]),
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2u])
    );
    hi = nbl_glsl_complex(
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*1u]),
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*3u])
    );
    */
    const uint tid = bitfieldReverse(gl_LocalInvocationIndex)>>23u;
    lo = nbl_glsl_complex(
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+0u]),
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2u])
    );
    hi = nbl_glsl_complex(
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+1u]),
        uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid+1u+_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_*2u])
    );
    barrier();
    if (is_inverse)
    {
        lo /= float(_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_<<1u);
        hi /= float(_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_<<1u);
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
 if (true)
 {
    nbl_glsl_complex values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD*2u]; // TODO: redo later
    // Load Values into local memory
    for(uint t=0u; t<item_per_thread_count; t++)
    {
        const uint tid = (t<<_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
        values[t] = nbl_glsl_ext_FFT_getPaddedData(nbl_glsl_ext_FFT_getCoordinates(tid),channel);
    }
#if 0
    // do huge FFT steps
    for (uint step=halfDataLength; step>_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_; step>>=1u)
    for(uint t=0u; t<virtual_thread_count; t++)
    {
        const uint pseudo_step = step>>_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_;
        const uint lo = t&(pseudo_step-1u);
        const uint lo_ix = nbl_glsl_bitfieldInsert_impl(t,0u,lo,1u);
        const uint hi_ix = lo_ix|pseudo_step;
        
        const uint subFFTItem = (lo<<_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
        nbl_glsl_FFT_DIF_radix2(nbl_glsl_FFT_twiddle(is_inverse,subFFTItem,float(step<<1u)),values[lo_ix],values[hi_ix]);
    }
#endif
    // do workgroup sized sub-FFTs
    for(uint t=0u; t<virtual_thread_count; t++)
    {
        const uint lo_ix = t<<1u;
        const uint hi_ix = lo_ix|1u;
        nbl_glsl_workgroup_FFT_DIF(is_inverse,values[lo_ix],values[hi_ix]);
    }
    // write out to main memory
    for(uint t=0u; t<item_per_thread_count; t++)
    {
        //const uint tid = (bitfieldReverse(t)>>(32u-findMSB(item_per_thread_count)-_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_))|gl_LocalInvocationIndex;
        const uint tid = (t<<_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_)|gl_LocalInvocationIndex;
        if (is_inverse)
            values[t] /= float(virtual_thread_count);
        nbl_glsl_ext_FFT_setData(nbl_glsl_ext_FFT_getCoordinates(tid),channel,values[t]);
    }
}
 else
 {
    const uint logTwo = findMSB(dataLength);
    uint thread_offset = gl_LocalInvocationIndex;

	// Pass 0: Bit Reversal
	uint leadingZeroes = nbl_glsl_clz(dataLength) + 1u;	
    nbl_glsl_complex even_values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD]; // should be half the prev version
    nbl_glsl_complex odd_values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD];

    // Load Initial Values into Local Mem (bit reversed indices)
    for(uint t = 0u; t<virtual_thread_count; t++)
    {
        uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;

        uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, 0, dataLength); // same as tid * 2
        uint odd_index = even_index + 1; 

        uvec3 coords_e = nbl_glsl_ext_FFT_getCoordinates(even_index);
        even_values[t] = nbl_glsl_ext_FFT_getPaddedData(coords_e, channel);

        uvec3 coords_o = nbl_glsl_ext_FFT_getCoordinates(odd_index);
        odd_values[t] = nbl_glsl_ext_FFT_getPaddedData(coords_o, channel);
    }

    // Initial Data Exchange
    {
        // Get Even/Odd Values X for virtual threads
        for(uint t = 0u; t<virtual_thread_count; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;
            
            uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, 0, dataLength); // same as tid * 2
            uint odd_index = even_index + 1;

            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index] = floatBitsToUint(even_values[t].x);
            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index] = floatBitsToUint(odd_values[t].x);
        }

        barrier();
        memoryBarrierShared();

        for(uint t = 0u; t<virtual_thread_count; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;

            uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, 0, dataLength); // same as tid * 2
            uint odd_index = even_index + 1;

            uint even_rev_bits = bitfieldReverse(even_index) >> leadingZeroes;
            uint odd_rev_bits = bitfieldReverse(odd_index) >> leadingZeroes;

            even_values[t].x = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_rev_bits]);
            odd_values[t].x = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_rev_bits]);
        }

        barrier();
        memoryBarrierShared();
        
        // Get Even/Odd Values Y for virtual threads
        for(uint t = 0u; t<virtual_thread_count; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;
            
            uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, 0, dataLength); // same as tid * 2
            uint odd_index = even_index + 1;

            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index] = floatBitsToUint(even_values[t].y);
            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index] = floatBitsToUint(odd_values[t].y);
        }

        barrier();
        memoryBarrierShared();

        for(uint t = 0u; t<virtual_thread_count; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;

            uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, 0, dataLength); // same as tid * 2
            uint odd_index = even_index + 1;

            uint even_rev_bits = bitfieldReverse(even_index) >> leadingZeroes;
            uint odd_rev_bits = bitfieldReverse(odd_index) >> leadingZeroes;

            even_values[t].y = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_rev_bits]);
            odd_values[t].y = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_rev_bits]);
        }
    }

    // For loop for each stage of the FFT (each virtual thread computes 1 buttefly)
	for(uint i = 0u; i < logTwo; ++i) 
    {
        // Computation of each virtual thread
        for(uint t = 0u; t<virtual_thread_count; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;
            const uint k = nbl_glsl_ext_FFT_calculateTwiddlePower(tid,i,logTwo);

            nbl_glsl_FFT_DIT_radix2(nbl_glsl_FFT_twiddle(is_inverse,k,logTwo),even_values[t],odd_values[t]);
        }

        // Exchange Even/Odd Values with Other Threads (or sometimes the same thread)
        if(i < logTwo - 1)
        {
            // Get Even/Odd Values X for virtual threads
            for(uint t = 0u; t<virtual_thread_count; t++)
            {
                uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;

                uint stage = i;
                uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, stage, dataLength);
                uint odd_index = even_index + (1u << stage);

                _NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index] = floatBitsToUint(even_values[t].x);
                _NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index] = floatBitsToUint(odd_values[t].x);
            }

            barrier();
            memoryBarrierShared();

            for(uint t = 0u; t<virtual_thread_count; t++)
            {
                uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;

                uint stage = i + 1u;
                uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, stage, dataLength);
                uint odd_index = even_index + (1u << stage);

                even_values[t].x = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index]);
                odd_values[t].x = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index]);
            }

            barrier();
            memoryBarrierShared();

            // Get Even/Odd Values Y for virtual threads
            for(uint t = 0u; t<virtual_thread_count; t++)
            {
                uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;

                uint stage = i;
                uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, stage, dataLength);
                uint odd_index = even_index + (1u << stage);

                _NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index] = floatBitsToUint(even_values[t].y);
                _NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index] = floatBitsToUint(odd_values[t].y);
            }

            barrier();
            memoryBarrierShared();

            for(uint t = 0u; t<virtual_thread_count; t++)
            {
                uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;

                uint stage = i + 1u;
                uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, stage, dataLength);
                uint odd_index = even_index + (1u << stage);

                even_values[t].y = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index]);
                odd_values[t].y = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index]);
            }
        }
    }
    
    for(uint t = 0u; t<virtual_thread_count; t++)
    {
        uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;

        uint stage = logTwo - 1;
        uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, stage, dataLength); // same as tid
        uint odd_index = even_index + (1u << stage);

	    uvec3 coords_e = nbl_glsl_ext_FFT_getCoordinates(even_index);
	    uvec3 coords_o = nbl_glsl_ext_FFT_getCoordinates(odd_index);

        nbl_glsl_complex complex_value_e = (!is_inverse) 
        ? even_values[t]
        : even_values[t] / dataLength;

        nbl_glsl_complex complex_value_o = (!is_inverse) 
        ? odd_values[t]
        : odd_values[t] / dataLength;

        nbl_glsl_ext_FFT_setData(coords_e, channel, complex_value_e);
        nbl_glsl_ext_FFT_setData(coords_o, channel, complex_value_o);
    }
}
}

#endif