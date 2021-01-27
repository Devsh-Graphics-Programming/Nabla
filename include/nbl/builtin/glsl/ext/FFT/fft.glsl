// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_INCLUDED_
#define _NBL_GLSL_EXT_FFT_INCLUDED_


// Shared Memory
#include <nbl/builtin/glsl/workgroup/shared_arithmetic.glsl>
#include <nbl/builtin/glsl/math/complex.glsl>

// TODO: Get from CPP Ext Side when Creating Shader
 #define _NBL_GLSL_EXT_FFT_SHARED_SIZE_NEEDED_ 256

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

#include <nbl/builtin/glsl/workgroup/shuffle.glsl>

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
float nbl_glsl_ext_FFT_getData(in uvec3 coordinate, in uint channel);
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
uint clz(uint x) 
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

uint reverseBits(uint x)
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

void nbl_glsl_ext_FFT(in nbl_glsl_ext_FFT_Uniforms_t inUniform)
{
	uint channel = 0;
    
	// Pass 0: Bit Reversal
	uint leadingZeroes = clz(inUniform.dimension.x) + 1;
	uint logTwo = 32 - leadingZeroes;
	
	uvec3 coords = uvec3(gl_LocalInvocationIndex.x, 0, 0);
	uint bitReversedIndex = reverseBits(coords.x) >> leadingZeroes;
	uvec3 bit_reversed_coords = uvec3(bitReversedIndex, 0, 0);

	float value = nbl_glsl_ext_FFT_getData(bit_reversed_coords, channel);

	float final_shuffled = value;
	for(uint i = 0; i < 1; ++i) {
		uint mask = 1 << i;
		float prev_shuffled = final_shuffled;
		final_shuffled = nbl_glsl_workgroupShuffleXor(prev_shuffled, mask);
	}

	vec2 complex_value = nbl_glsl_eITheta(3.0 / 4.0 * nbl_glsl_PI); // = vec2(coords, final_shuffled);
	nbl_glsl_ext_FFT_setData(coords, channel, complex_value);
}

#endif