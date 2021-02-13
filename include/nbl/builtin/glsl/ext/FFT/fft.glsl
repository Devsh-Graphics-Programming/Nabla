// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_INCLUDED_
#define _NBL_GLSL_EXT_FFT_INCLUDED_

#include <nbl/builtin/glsl/math/complex.glsl>

// Shared Memory
#include <nbl/builtin/glsl/workgroup/shared_arithmetic.glsl>


#ifndef _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_
#error "_NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ should be defined."
#endif

#ifndef _NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD
#error "_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD should be defined."
#endif


// TODO: Investigate why +1 solves all glitches
#define _NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_ _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ + 1

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

#ifndef _NBL_GLSL_EXT_FFT_PUSH_CONSTANTS_DEFINED_
#define _NBL_GLSL_EXT_FFT_PUSH_CONSTANTS_DEFINED_
layout(push_constant) uniform PushConstants
{
    layout (offset = 0) uvec3 dimension;
    layout (offset = 16) uvec3 padded_dimension;
	layout (offset = 32) uint direction;
    layout (offset = 36) uint is_inverse;
    layout (offset = 40) uint padding_type; // clamp_to_edge or fill_with_zero
} pc;
#endif

#ifndef _NBL_GLSL_EXT_FFT_GET_DATA_DECLARED_
#define _NBL_GLSL_EXT_FFT_GET_DATA_DECLARED_
vec2 nbl_glsl_ext_FFT_getData(in uvec3 coordinate, in uint channel);
#endif

#ifndef _NBL_GLSL_EXT_FFT_SET_DATA_DECLARED_
#define _NBL_GLSL_EXT_FFT_SET_DATA_DECLARED_
void nbl_glsl_ext_FFT_setData(in uvec3 coordinate, in uint channel, in vec2 complex_value);
#endif

#ifndef _NBL_GLSL_EXT_FFT_GET_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_FFT_getData` and mark `_NBL_GLSL_EXT_FFT_GET_DATA_DEFINED_`!"
#endif
#ifndef _NBL_GLSL_EXT_FFT_SET_DATA_DEFINED_
#error "You need to define `nbl_glsl_ext_FFT_setData` and mark `_NBL_GLSL_EXT_FFT_SET_DATA_DEFINED_`!"
#endif

// Count Leading Zeroes (naive?)
uint nbl_glsl_ext_FFT_clz(in uint x) 
{
    return 31 - findMSB(x);
}

uint nbl_glsl_ext_FFT_reverseBits(in uint x)
{
    return bitfieldReverse(x);
}

uint nbl_glsl_ext_FFT_calculateTwiddlePower(in uint threadId, in uint iteration, in uint logTwoN, in uint N) 
{
    return (threadId & ((N / (1u << (logTwoN - iteration))) * 2 - 1)) * ((1u << (logTwoN - iteration)) / 2);;
}

vec2 nbl_glsl_ext_FFT_twiddle(in uint threadId, in uint iteration, in uint logTwoN, in uint N) 
{
    uint k = nbl_glsl_ext_FFT_calculateTwiddlePower(threadId, iteration, logTwoN, N);
    return nbl_glsl_eITheta(-1 * 2 * nbl_glsl_PI * k / N);
}

vec2 nbl_gnbl_glsl_ext_FFT_twiddleInverse(in uint threadId, in uint iteration, in uint logTwoN, in uint N) 
{
    return nbl_glsl_complex_conjugate(nbl_glsl_ext_FFT_twiddle(threadId, iteration, logTwoN, N));
}

uint nbl_glsl_ext_FFT_getChannel()
{
    return gl_WorkGroupID[pc.direction];
}

uvec3 nbl_glsl_ext_FFT_getCoordinates(in uint tidx)
{
    uvec3 tmp = gl_WorkGroupID;
    tmp[pc.direction] = tidx;
    return tmp;
}

uvec3 nbl_glsl_ext_FFT_getBitReversedCoordinates(in uvec3 coords, in uint leadingZeroes)
{
    uint bitReversedIndex = nbl_glsl_ext_FFT_reverseBits(coords[pc.direction]) >> leadingZeroes;
    uvec3 tmp = coords;
    tmp[pc.direction] = bitReversedIndex;
    return tmp;
}

uint nbl_glsl_ext_FFT_getDimLength(uvec3 dimension)
{
    uint dataLength = 0;

    if(pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_X_) {
        return dimension.x;
    } else if (pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_Y_) {
        return dimension.y;
    } else if (pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_Z_) {
        return dimension.z;
    }

    return dataLength;
}

vec2 nbl_glsl_ext_FFT_getPaddedData(in uvec3 coordinate, in uint channel) {
    uint min_x = 0;
    uint max_x = pc.dimension.x + min_x - 1;

    uint min_y = 0;
    uint max_y = pc.dimension.y + min_y - 1;

    uint min_z = 0;
    uint max_z = pc.dimension.z + min_z - 1;


    uvec3 actual_coord = uvec3(0, 0, 0);

    if(_NBL_GLSL_EXT_FFT_CLAMP_TO_EDGE_ == pc.padding_type) {
        if (coordinate.x < min_x) {
            actual_coord.x = 0;
        } else if(coordinate.x > max_x) {
            actual_coord.x = pc.dimension.x - 1;
        } else {
            actual_coord.x = coordinate.x - min_x;
        }
        
        if (coordinate.y < min_y) {
            actual_coord.y = 0;
        } else if (coordinate.y > max_y) {
            actual_coord.y = pc.dimension.y - 1;
        } else {
            actual_coord.y = coordinate.y - min_y;
        }
        
        if (coordinate.z < min_z) {
            actual_coord.z = 0;
        } else if (coordinate.z > max_z) {
            actual_coord.z = pc.dimension.z - 1;
        } else {
            actual_coord.z = coordinate.z - min_z;
        }
        
    } else if (_NBL_GLSL_EXT_FFT_FILL_WITH_ZERO_ == pc.padding_type) {

        if ( coordinate.x < min_x || coordinate.x > max_x ||
             coordinate.y < min_y || coordinate.y > max_y ||
             coordinate.z < min_z || coordinate.z > max_z ) {
            return vec2(0, 0);
        }
        
        actual_coord.x = coordinate.x - min_x;
        actual_coord.y = coordinate.y - min_y;
        actual_coord.z = coordinate.z - min_z;

    }
    
    return nbl_glsl_ext_FFT_getData(actual_coord, channel);
}

void nbl_glsl_ext_FFT()
{
    // Virtual Threads Calculation
    uint dataLength = nbl_glsl_ext_FFT_getDimLength(pc.padded_dimension);
    uint num_virtual_threads = uint(ceil(float(dataLength) / float(_NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_)));
    uint thread_offset = gl_LocalInvocationIndex;

	uint channel = nbl_glsl_ext_FFT_getChannel();
    
	// Pass 0: Bit Reversal
	uint leadingZeroes = nbl_glsl_ext_FFT_clz(dataLength) + 1;
	uint logTwo = 32 - leadingZeroes;
	
    vec2 current_values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD];
    vec2 shuffled_values[_NBL_GLSL_EXT_FFT_MAX_ITEMS_PER_THREAD];

    // Load Initial Values into Local Mem (bit reversed indices)
    for(uint t = 0; t < num_virtual_threads; t++)
    {
        uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_;
        uvec3 coords = nbl_glsl_ext_FFT_getCoordinates(tid);
        uvec3 bitReversedCoords = nbl_glsl_ext_FFT_getBitReversedCoordinates(coords, leadingZeroes);

        current_values[t] = nbl_glsl_ext_FFT_getPaddedData(bitReversedCoords, channel);
    }

    // For loop for each stage of the FFT (each virtual thread computes 1 buttefly wing)
	for(uint i = 0; i < logTwo; ++i) 
    {
		uint mask = 1 << i;

        // Data Exchange for virtual threads :
        // X and Y are seperate to use less shared memory for complex numbers
        // Get Shuffled Values X for virtual threads
        for(uint t = 0; t < num_virtual_threads; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_;
            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid] = floatBitsToUint(current_values[t].x);
        }
        barrier();
        memoryBarrierShared();
        for(uint t = 0; t < num_virtual_threads; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_;
            shuffled_values[t].x = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid ^ mask]);
        }

        // Get Shuffled Values Y for virtual threads
        for(uint t = 0; t < num_virtual_threads; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_;
            _NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid] = floatBitsToUint(current_values[t].y);
        }
        barrier();
        memoryBarrierShared();
        for(uint t = 0; t < num_virtual_threads; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_;
            shuffled_values[t].y = uintBitsToFloat(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[tid ^ mask]);
        }

        // Computation of each virtual thread
        for(uint t = 0; t < num_virtual_threads; t++)
        {
            uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_;
            vec2 shuffled_value = shuffled_values[t];

            vec2 nbl_glsl_ext_FFT_twiddle = (0 == pc.is_inverse) 
             ? nbl_glsl_ext_FFT_twiddle(tid, i, logTwo, dataLength)
             : nbl_gnbl_glsl_ext_FFT_twiddleInverse(tid, i, logTwo, dataLength);

            vec2 this_value = current_values[t];

            if(0 < uint(tid & mask)) {
                current_values[t] = shuffled_value + nbl_glsl_complex_mul(nbl_glsl_ext_FFT_twiddle, this_value); 
            } else {
                current_values[t] = this_value + nbl_glsl_complex_mul(nbl_glsl_ext_FFT_twiddle, shuffled_value); 
            }
        }
    }

    for(uint t = 0; t < num_virtual_threads; t++)
    {
        uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_;
	    uvec3 coords = nbl_glsl_ext_FFT_getCoordinates(tid);
        vec2 complex_value = (0 == pc.is_inverse) 
         ? current_values[t]
         : current_values[t] / dataLength;

	    nbl_glsl_ext_FFT_setData(coords, channel, complex_value);
    }
}

#endif