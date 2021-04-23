// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_GLSL_EXT_FFT_TYPES_INCLUDED_
#define _NBL_GLSL_EXT_FFT_TYPES_INCLUDED_

#include <nbl/builtin/glsl/math/complex.glsl>

#if _NBL_GLSL_EXT_FFT_HALF_STORAGE_!=0
#define nbl_glsl_ext_FFT_storage_t uint

nbl_glsl_complex nbl_glsl_ext_FFT_storage_t_get(in nbl_glsl_ext_FFT_storage_t _in)
{
	return unpackHalf2x16(_in);
}
void nbl_glsl_ext_FFT_storage_t_set(out nbl_glsl_ext_FFT_storage_t _out, in nbl_glsl_complex _in)
{
	_out = packHalf2x16(_in);
}
#else
#define nbl_glsl_ext_FFT_storage_t nbl_glsl_complex

nbl_glsl_complex nbl_glsl_ext_FFT_storage_t_get(in nbl_glsl_ext_FFT_storage_t _in)
{
	return _in;
}
void nbl_glsl_ext_FFT_storage_t_set(out nbl_glsl_ext_FFT_storage_t _out, in nbl_glsl_complex _in)
{
	_out = _in;
}
#endif

#include <nbl/builtin/glsl/ext/FFT/parameters.glsl>

#endif