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


void nbl_glsl_ext_FFT()
{
	uvec3 coords = uvec3(gl_GlobalInvocationID.x, 0, 0);
	uint channel = 0;

	float value = nbl_glsl_ext_FFT_getData(coords, channel);
	vec2 complex_value = vec2(value, float(gl_LocalInvocationID));
	nbl_glsl_ext_FFT_setData(coords, channel, complex_value);
}

#endif