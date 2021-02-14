// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_EXT_FFT_PARAMETERS_INCLUDED_
#define _NBL_GLSL_EXT_FFT_PARAMETERS_INCLUDED_

struct nbl_glsl_ext_FFT_Parameters_t
{
    uvec3   dimension;
	uint    direction_isInverse_paddingType; // packed into a uint
    uvec3   padded_dimension;
};

#endif