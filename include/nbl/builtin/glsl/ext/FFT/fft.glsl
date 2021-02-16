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

#ifndef _NBL_GLSL_EXT_FFT_GET_PARAMETERS_DECLARED_
#define _NBL_GLSL_EXT_FFT_GET_PARAMETERS_DECLARED_
nbl_glsl_ext_FFT_Parameters_t nbl_glsl_ext_FFT_getParameters();
#endif

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

uvec3 nbl_glsl_ext_FFT_Parameters_t_getPaddedDimensions() {
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return (params.padded_dimension.xyz);
}
uvec3 nbl_glsl_ext_FFT_Parameters_t_getDimensions() {
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return (params.dimension.xyz);
}
uint nbl_glsl_ext_FFT_Parameters_t_getDirection() {
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return (params.dimension.w >> 16) & 0x000000ff;
}
bool nbl_glsl_ext_FFT_Parameters_t_getIsInverse() {
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return bool((params.dimension.w >> 8) & 0x000000ff);
}
uint nbl_glsl_ext_FFT_Parameters_t_getPaddingType() {
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return (params.dimension.w) & 0x000000ff;
}
uint nbl_glsl_ext_FFT_Parameters_t_getNumChannels() {
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return (params.padded_dimension.w);
}

uvec3 nbl_glsl_ext_FFT_getCoordinates(in uint tidx)
{
    uint direction = nbl_glsl_ext_FFT_Parameters_t_getDirection();
    uvec3 tmp = gl_WorkGroupID;
    tmp[direction] = tidx;
    return tmp;
}

uvec3 nbl_glsl_ext_FFT_getBitReversedCoordinates(in uvec3 coords, in uint leadingZeroes)
{
    uint direction = nbl_glsl_ext_FFT_Parameters_t_getDirection();
    uint bitReversedIndex = bitfieldReverse(coords[direction]) >> leadingZeroes;
    uvec3 tmp = coords;
    tmp[direction] = bitReversedIndex;
    return tmp;
}

uint nbl_glsl_ext_FFT_getDimLength(uvec3 dimension)
{
    uint direction = nbl_glsl_ext_FFT_Parameters_t_getDirection();
    return dimension[direction];
}

uint nbl_glsl_ext_FFT_calculateTwiddlePower(in uint threadId, in uint iteration, in uint logTwoN) 
{
    const uint shiftSuffix = logTwoN - 1u - iteration;
    const uint suffixMask = (1u << iteration) - 1u;
    return (threadId & suffixMask) << shiftSuffix;
}

nbl_glsl_complex nbl_glsl_ext_FFT_twiddle(in uint threadId, in uint iteration, in uint logTwoN) 
{
    uint k = nbl_glsl_ext_FFT_calculateTwiddlePower(threadId, iteration, logTwoN);
    return nbl_glsl_expImaginary(-1.0f * 2.0f * nbl_glsl_PI * float(k) / float(1 << logTwoN));
}

nbl_glsl_complex nbl_glsl_ext_FFT_twiddleInverse(in uint threadId, in uint iteration, in uint logTwoN) 
{
    return nbl_glsl_complex_conjugate(nbl_glsl_ext_FFT_twiddle(threadId, iteration, logTwoN));
}

uint nbl_glsl_ext_FFT_getEvenIndex(in uint threadId, in uint iteration, in uint N) {
    return ((threadId & (N - (1u << iteration))) << 1u) | (threadId & ((1u << iteration) - 1u));
}

void nbl_glsl_ext_FFT(bool is_inverse, uint channel)
{
    // Virtual Threads Calculation
    uint dataLength = nbl_glsl_ext_FFT_getDimLength(nbl_glsl_ext_FFT_Parameters_t_getPaddedDimensions());
    uint num_virtual_threads = ((dataLength >> 1)-1u)/(_NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_)+1u;
    uint thread_offset = gl_LocalInvocationIndex;

	// Pass 0: Bit Reversal
	uint leadingZeroes = nbl_glsl_clz(dataLength) + 1u;
	uint logTwo = 32u - leadingZeroes;
	
    nbl_glsl_complex even_values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD]; // should be half the prev version
    nbl_glsl_complex odd_values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD];

    // Load Initial Values into Local Mem (bit reversed indices)
    for(uint t = 0u; t < num_virtual_threads; t++)
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
        for(uint t = 0u; t < num_virtual_threads; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;
            
            uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, 0, dataLength); // same as tid * 2
            uint odd_index = even_index + 1;

            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index] = floatBitsToUint(even_values[t].x);
            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index] = floatBitsToUint(odd_values[t].x);
        }

        barrier();
        memoryBarrierShared();

        for(uint t = 0u; t < num_virtual_threads; t++)
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
        for(uint t = 0u; t < num_virtual_threads; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;
            
            uint even_index = nbl_glsl_ext_FFT_getEvenIndex(tid, 0, dataLength); // same as tid * 2
            uint odd_index = even_index + 1;

            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[even_index] = floatBitsToUint(even_values[t].y);
            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[odd_index] = floatBitsToUint(odd_values[t].y);
        }

        barrier();
        memoryBarrierShared();

        for(uint t = 0u; t < num_virtual_threads; t++)
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
        for(uint t = 0u; t < num_virtual_threads; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_WORKGROUP_SIZE_;
            nbl_glsl_complex even_value = even_values[t];
            nbl_glsl_complex odd_value = odd_values[t];

            nbl_glsl_complex twiddle = (!is_inverse) 
            ? nbl_glsl_ext_FFT_twiddle(tid, i, logTwo)
            : nbl_glsl_ext_FFT_twiddleInverse(tid, i, logTwo);

            nbl_glsl_complex cmplx_mul = nbl_glsl_complex_mul(twiddle, odd_value);

            even_values[t] = even_value + cmplx_mul; 
            odd_values[t] = even_value - cmplx_mul; 
        }

        // Exchange Even/Odd Values with Other Threads (or sometimes the same thread)
        if(i < logTwo - 1)
        {
            // Get Even/Odd Values X for virtual threads
            for(uint t = 0u; t < num_virtual_threads; t++)
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

            for(uint t = 0u; t < num_virtual_threads; t++)
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
            for(uint t = 0u; t < num_virtual_threads; t++)
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

            for(uint t = 0u; t < num_virtual_threads; t++)
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
    
    for(uint t = 0u; t < num_virtual_threads; t++)
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

#endif