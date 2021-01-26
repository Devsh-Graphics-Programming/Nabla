// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_INCLUDED_
#define _NBL_GLSL_EXT_FFT_INCLUDED_

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
	
	uvec3 coords = uvec3(gl_GlobalInvocationID.x, 0, 0);

	// Pass 0: Bit Reversal
	uint leadingZeroes = clz(inUniform.dimension.x) + 1;
	uint logTwo = 32 - leadingZeroes;

	uint bitReversed = reverseBits(coords.x) >> leadingZeroes;

	float value = nbl_glsl_ext_FFT_getData(coords, channel);

	vec2 complex_value = vec2(value, float(bitReversed));

	nbl_glsl_ext_FFT_setData(coords, channel, complex_value);
}

#endif