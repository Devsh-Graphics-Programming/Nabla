// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_INCLUDED_
#define _NBL_GLSL_EXT_FFT_INCLUDED_

#include <nbl/builtin/glsl/math/complex.glsl>

// Shared Memory
#include <nbl/builtin/glsl/workgroup/shared_arithmetic.glsl>

// TODO: These are very badly implemented and not completely correct. Fix later and get from CPP extension when creating shader
 #define _NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ 512
 #define _NBL_GLSL_EXT_FFT_LOCAL_DATA_SIZE NBL_GLSL_EVAL(_NBL_GLSL_EXT_FFT_MAX_DIM_SIZE_ / _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_)
 #define _NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_ 1024



#ifdef _NBL_GLSL_SCRATCH_SHARED_DEFINED_
    #if NBL_GLSL_LESS(_NBL_GLSL_SCRATCH_SHARED_SIZE_DEFINED_,_NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_)
        #error "Not enough shared memory declared for ext::FFT!"
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

#ifndef _NBL_GLSL_EXT_FFT_PUSH_CONSTANTS_DEFINED_
#define _NBL_GLSL_EXT_FFT_PUSH_CONSTANTS_DEFINED_
layout(push_constant) uniform PushConstants
{
	uint direction;
} pc;
#endif

 // Uniform
#ifndef _NBL_GLSL_EXT_FFT_UNIFORMS_DEFINED_
#define _NBL_GLSL_EXT_FFT_UNIFORMS_DEFINED_
struct nbl_glsl_ext_FFT_Uniforms_t
{
	uvec3 dimension;
};
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
uint clz(in uint x) 
{
    uint n = 0;
	if (x == 0) { return 32; }
    if (x <= 0x0000ffff) { n += 16; x <<= 16; }
    if (x <= 0x00ffffff) { n +=  8; x <<= 8; }
    if (x <= 0x0fffffff) { n +=  4; x <<= 4; }
    if (x <= 0x3fffffff) { n +=  2; x <<= 2; }
    if (x <= 0x7fffffff) { n++; };
    return n;
}

uint reverseBits(in uint x)
{
	uint count = 4 * 8 - 1;
	uint reverse_num = x; 

    x >>= 1;  
    while(x > 0) 
    { 
       reverse_num <<= 1;        
       reverse_num |= x & 1; 
       x >>= 1; 
       count--; 
    } 
    reverse_num <<= count;
    return reverse_num; 
}

uint calculate_twiddle_power(in uint threadId, in uint iteration, in uint logTwoN, in uint N) 
{
    return (threadId & ((N / (1u << (logTwoN - iteration))) * 2 - 1)) * ((1u << (logTwoN - iteration)) / 2);;
}

vec2 twiddle(in uint threadId, in uint iteration, in uint logTwoN, in uint N) 
{
    uint k = calculate_twiddle_power(threadId, iteration, logTwoN, N);
    return nbl_glsl_eITheta(-1 * 2 * nbl_glsl_PI * k / N);
}

vec2 twiddle_inv(in uint threadId, in uint iteration, in uint logTwoN, in uint N) 
{
    float k = calculate_twiddle_power(threadId, iteration, logTwoN, N);
    return nbl_glsl_eITheta(2 * nbl_glsl_PI * k / N);
}

uint getChannel()
{
    if(pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_X_) {
        return gl_WorkGroupID.x;
    } else if (pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_Y_) {
        return gl_WorkGroupID.y;
    } else if (pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_Z_) {
        return gl_WorkGroupID.z;
    } else {
        return 0;
    }
}

uvec3 getCoordinates(in uint tidx)
{
    if(pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_X_) {
        return uvec3(tidx, gl_WorkGroupID.y, gl_WorkGroupID.z);
    } else if (pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_Y_) {
        return uvec3(gl_WorkGroupID.x, tidx, gl_WorkGroupID.z);
    } else if (pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_Z_) {
        return uvec3(gl_WorkGroupID.x, gl_WorkGroupID.y, tidx);
    } else {
        return uvec3(0,0,0);
    }
}

uvec3 getBitReversedCoordinates(in uvec3 coords, in uint leadingZeroes)
{
    if(pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_X_) {
        uint bitReversedIndex = reverseBits(coords.x) >> leadingZeroes;
        return uvec3(bitReversedIndex, coords.y, coords.z);
    } else if (pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_Y_) {
        uint bitReversedIndex = reverseBits(coords.y) >> leadingZeroes;
        return uvec3(coords.x, bitReversedIndex, coords.z);
    } else if (pc.direction == _NBL_GLSL_EXT_FFT_DIRECTION_Z_) {
        uint bitReversedIndex = reverseBits(coords.z) >> leadingZeroes;
        return uvec3(coords.x, coords.y, bitReversedIndex);
    } else {
        return uvec3(0,0,0);
    }
}

uint getDimLength(uvec3 dimension)
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

void nbl_glsl_ext_FFT(in nbl_glsl_ext_FFT_Uniforms_t inUniform, const bool is_inverse)
{
    // Virtual Threads Calculation
    uint dataLength = getDimLength(inUniform.dimension);
    uint num_virtual_threads = uint(ceil(float(dataLength) / float(_NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_)));
    uint thread_offset = gl_LocalInvocationIndex;

	uint channel = getChannel();
    
	// Pass 0: Bit Reversal
	uint leadingZeroes = clz(dataLength) + 1;
	uint logTwo = 32 - leadingZeroes;
	
    vec2 current_values[_NBL_GLSL_EXT_FFT_LOCAL_DATA_SIZE];
    vec2 shuffled_values[_NBL_GLSL_EXT_FFT_LOCAL_DATA_SIZE];

    // Load Initial Values into Local Mem (bit reversed indices)
    for(uint t = 0; t < num_virtual_threads; t++)
    {
        uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_;
        uvec3 coords = getCoordinates(tid);
        uvec3 bitReversedCoords = getBitReversedCoordinates(coords, leadingZeroes);

        current_values[t] = nbl_glsl_ext_FFT_getData(bitReversedCoords, channel);
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

            vec2 twiddle = (!is_inverse) 
             ? twiddle(tid, i, logTwo, dataLength)
             : twiddle_inv(tid, i, logTwo, dataLength);

            vec2 prev_value = current_values[t];
            current_values[t] = shuffled_value + nbl_glsl_complex_mul(twiddle, prev_value); 
        }
    }

    for(uint t = 0; t < num_virtual_threads; t++)
    {
        uint tid = thread_offset + t * _NBL_GLSL_EXT_FFT_BLOCK_SIZE_X_DEFINED_;
	    uvec3 coords = getCoordinates(tid);
        vec2 complex_value = (!is_inverse) 
         ? current_values[t]
         : current_values[t] / dataLength;

	    nbl_glsl_ext_FFT_setData(coords, channel, complex_value);
    }
}

#endif