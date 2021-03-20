// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_PARAMETERS_INCLUDED_
#define _NBL_GLSL_EXT_FFT_PARAMETERS_INCLUDED_

#include "nbl/builtin/glsl/ext/FFT/parameters_struct.glsl"


#define _NBL_GLSL_EXT_FFT_DIRECTION_X_ 0
#define _NBL_GLSL_EXT_FFT_DIRECTION_Y_ 1
#define _NBL_GLSL_EXT_FFT_DIRECTION_Z_ 2

#define _NBL_GLSL_EXT_FFT_CLAMP_TO_EDGE_ 0
#define _NBL_GLSL_EXT_FFT_FILL_WITH_ZERO_ 1


#ifndef _NBL_GLSL_EXT_FFT_GET_PARAMETERS_DECLARED_
#define _NBL_GLSL_EXT_FFT_GET_PARAMETERS_DECLARED_
nbl_glsl_ext_FFT_Parameters_t nbl_glsl_ext_FFT_getParameters();
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

uint nbl_glsl_ext_FFT_Parameters_t_getFFTLength() {
    const uint direction = nbl_glsl_ext_FFT_Parameters_t_getDirection();
    return nbl_glsl_ext_FFT_Parameters_t_getPaddedDimensions()[direction];
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

#endif