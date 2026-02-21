// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_PARAMETERS_INCLUDED_
#define _NBL_GLSL_EXT_FFT_PARAMETERS_INCLUDED_

#include "nbl/builtin/glsl/ext/FFT/parameters_struct.glsl"


#define _NBL_GLSL_EXT_FFT_DIRECTION_X_ 0
#define _NBL_GLSL_EXT_FFT_DIRECTION_Y_ 1
#define _NBL_GLSL_EXT_FFT_DIRECTION_Z_ 2

#define _NBL_GLSL_EXT_FFT_PAD_REPEAT_ 0
#define _NBL_GLSL_EXT_FFT_PAD_CLAMP_TO_EDGE_ 1
#define _NBL_GLSL_EXT_FFT_PAD_CLAMP_TO_BORDER_ 2
#define _NBL_GLSL_EXT_FFT_PAD_MIRROR_ 3


#ifndef _NBL_GLSL_EXT_FFT_GET_PARAMETERS_DECLARED_
#define _NBL_GLSL_EXT_FFT_GET_PARAMETERS_DECLARED_
nbl_glsl_ext_FFT_Parameters_t nbl_glsl_ext_FFT_getParameters();
#endif


#ifndef _NBL_GLSL_EXT_FFT_PARAMETERS_METHODS_DECLARED_
#define _NBL_GLSL_EXT_FFT_PARAMETERS_METHODS_DECLARED_
uvec3 nbl_glsl_ext_FFT_Parameters_t_getDimensions()
{
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return params.input_dimensions.xyz;
}

bool nbl_glsl_ext_FFT_Parameters_t_getIsInverse()
{
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return bool(params.input_dimensions.w>>31u);
}
uint nbl_glsl_ext_FFT_Parameters_t_getDirection()
{
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return (params.input_dimensions.w>>28u)&0x3u;
}
uint nbl_glsl_ext_FFT_Parameters_t_getMaxChannel()
{
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return (params.input_dimensions.w>>26u)&0x3u;
}
uint nbl_glsl_ext_FFT_Parameters_t_getLog2FFTSize()
{
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return (params.input_dimensions.w>>3u)&0x1fu;
}
uint nbl_glsl_ext_FFT_Parameters_t_getPaddingType()
{
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return params.input_dimensions.w&0x7u;
}

uvec4 nbl_glsl_ext_FFT_Parameters_t_getInputStrides()
{
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return params.input_strides;
}

uvec4 nbl_glsl_ext_FFT_Parameters_t_getOutputStrides()
{
    nbl_glsl_ext_FFT_Parameters_t params = nbl_glsl_ext_FFT_getParameters();
    return params.output_strides;
}
#endif

#endif