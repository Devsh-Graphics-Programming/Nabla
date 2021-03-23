// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/builtin/glsl/ext/FFT/parameters_struct.glsl"
struct convolve_parameters_t
{
    nbl_glsl_ext_FFT_Parameters_t fft_params;
    vec2    bitreversed_to_normalized;
    vec2    kernel_half_pixel_size;
};