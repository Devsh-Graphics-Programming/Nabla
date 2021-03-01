// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_INCLUDED_
#define _NBL_GLSL_EXT_FFT_INCLUDED_

#include <nbl/builtin/glsl/math/complex.glsl>

// Shared Memory
#include <nbl/builtin/glsl/workgroup/shared_arithmetic.glsl>
#include <nbl/builtin/glsl/math/functions.glsl>
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

void nbl_glsl_ext_FFT(bool is_inverse, uint channel)
{
    // Virtual Threads Calculation
    const uint dataLength = nbl_glsl_ext_FFT_Parameters_t_getFFTLength();
    const uint logTwo = findMSB(dataLength);

    const uint halfDataLength = dataLength>>1u;
    const uint last_virtual_thread = (halfDataLength-1u)>>_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_;
#if 0
    nbl_glsl_complex values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD];
    // Load Values into local memory
    for(uint t=0u; t<=last_virtual_thread; t++)
    {
        const uint tid = gl_LocalInvocationIndex+(t<<_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_);
        values[t] = nbl_glsl_ext_FFT_getPaddedData(lower_coord,channel);
    }
    // do huge FFT steps
    for (uint step=halfDataLength>>_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_LOG2_; step!=0u; --step)
    {
        nbl_glsl_complex even = values[t];
        nbl_glsl_complex odd = values[t+step];
        
        const uint k = nbl_glsl_ext_FFT_calculateTwiddlePower(tid, i, logTwo);
        nbl_glsl_complex twiddle = nbl_glsl_FFT_twiddle(is_inverse, tid, k, logTwo);

        values[t] = even+odd;
        values[t+step] = nbl_glsl_complex_mul(even-odd,);
    }
    for (uint tid=0u; tid<halfDataLength; tid+=_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_)
    {
        const uvec3 lower_coord = nbl_glsl_ext_FFT_getCoordinates(tid);
        const uvec3 higher_coord = nbl_glsl_ext_FFT_getCoordinates(tid+halfDataLength);

        const nbl_glsl_complex lo = nbl_glsl_ext_FFT_getPaddedData(lower_coord,channel);
        const nbl_glsl_complex hi = nbl_glsl_ext_FFT_getPaddedData(higher_coord,channel);
        const uint k = nbl_glsl_ext_FFT_calculateTwiddlePower(tid, i, logTwo);
        nbl_glsl_complex twiddle = nbl_glsl_FFT_twiddle(is_inverse, tid, k, logTwo);
        const nbl_glsl_complex out_lo = lo+hi;
        const nbl_glsl_complex out_hi = nbl_glsl_complex_mul(lo-hi);

        nbl_glsl_ext_FFT_setData(lower_coord, channel, out_lo);
        nbl_glsl_ext_FFT_setData(higher_coord, channel, out_hi);
    }
#else
    uint thread_offset = gl_LocalInvocationIndex;

	// Pass 0: Bit Reversal
	uint leadingZeroes = nbl_glsl_clz(dataLength) + 1u;	
    nbl_glsl_complex even_values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD]; // should be half the prev version
    nbl_glsl_complex odd_values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD];

    // Load Initial Values into Local Mem (bit reversed indices)
    for(uint t = 0u; t<=last_virtual_thread; t++)
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
        for(uint t = 0u; t<=last_virtual_thread; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;
            
            uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, 0, dataLength); // same as tid * 2
            uint odd_index = even_index + 1;

            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index] = floatBitsToUint(even_values[t].x);
            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index] = floatBitsToUint(odd_values[t].x);
        }

        barrier();
        memoryBarrierShared();

        for(uint t = 0u; t<=last_virtual_thread; t++)
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
        for(uint t = 0u; t<=last_virtual_thread; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;
            
            uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, 0, dataLength); // same as tid * 2
            uint odd_index = even_index + 1;

            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index] = floatBitsToUint(even_values[t].y);
            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index] = floatBitsToUint(odd_values[t].y);
        }

        barrier();
        memoryBarrierShared();

        for(uint t = 0u; t<=last_virtual_thread; t++)
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
        for(uint t = 0u; t<=last_virtual_thread; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;
            const uint k = nbl_glsl_ext_FFT_calculateTwiddlePower(tid,i,logTwo);

            nbl_glsl_FFT_DIT_radix2(nbl_glsl_FFT_twiddle(is_inverse,k,logTwo),even_values[t],odd_values[t]);
        }

        // Exchange Even/Odd Values with Other Threads (or sometimes the same thread)
        if(i < logTwo - 1)
        {
            // Get Even/Odd Values X for virtual threads
            for(uint t = 0u; t<=last_virtual_thread; t++)
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

            for(uint t = 0u; t<=last_virtual_thread; t++)
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
            for(uint t = 0u; t<=last_virtual_thread; t++)
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

            for(uint t = 0u; t<=last_virtual_thread; t++)
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
    
    for(uint t = 0u; t<=last_virtual_thread; t++)
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
#endif
}

#endif