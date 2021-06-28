// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/glsl/ext/FFT/parameters_struct.glsl"
struct convolve_parameters_t
{
    nbl_glsl_ext_FFT_Parameters_t fft;
    vec2    kernel_half_pixel_size;
};

struct image_store_parameters_t
{
    nbl_glsl_ext_FFT_Parameters_t fft;
    ivec2   unpad_offset;
};