// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_PARAMETERS_STRUCT_INCLUDED_
#define _NBL_GLSL_EXT_FFT_PARAMETERS_STRUCT_INCLUDED_

struct nbl_glsl_ext_FFT_Parameters_t
{
    uvec4   input_dimensions; // settings packed into the w component : (isInverse << 31u) | (direction_u3 << 28u) | (maxChannel_u2<<26u) | (fftSizeLog2_u5 << 3u) | paddingType_u3;
    uvec4   input_strides;
    uvec4   output_strides;
};

#endif