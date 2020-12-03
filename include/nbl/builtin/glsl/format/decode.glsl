// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_FORMAT_DECODE_INCLUDED_
#define _NBL_BUILTIN_GLSL_FORMAT_DECODE_INCLUDED_

#include <nbl/builtin/glsl/format/constants.glsl>

vec3 nbl_glsl_decodeRGB19E7(in uvec2 x)
{
	int exp = int(bitfieldExtract(x.y, 3*nbl_glsl_RGB19E7_MANTISSA_BITS-32, nbl_glsl_RGB19E7_EXPONENT_BITS) - nbl_glsl_RGB19E7_EXP_BIAS - nbl_glsl_RGB19E7_MANTISSA_BITS);
	float scale = exp2(float(exp));//uintBitsToFloat((uint(exp)+127u)<<23u)
	
	vec3 v;
	v.x = int(bitfieldExtract(x.x, 0, nbl_glsl_RGB19E7_MANTISSA_BITS))*scale;
	v.y = int(
		bitfieldExtract(x.x, nbl_glsl_RGB19E7_MANTISSA_BITS, 32-nbl_glsl_RGB19E7_MANTISSA_BITS) | 
		(bitfieldExtract(x.y, 0, nbl_glsl_RGB19E7_MANTISSA_BITS-(32-nbl_glsl_RGB19E7_MANTISSA_BITS))<<(32-nbl_glsl_RGB19E7_MANTISSA_BITS))
	) * scale;
	v.z = int(bitfieldExtract(x.y, nbl_glsl_RGB19E7_MANTISSA_BITS-(32-nbl_glsl_RGB19E7_MANTISSA_BITS), nbl_glsl_RGB19E7_MANTISSA_BITS)) * scale;
	
	return v;
}

#endif