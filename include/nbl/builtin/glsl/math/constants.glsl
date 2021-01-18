// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_MATH_CONSTANTS_INCLUDED_
#define _NBL_MATH_CONSTANTS_INCLUDED_

#include <nbl/builtin/glsl/limits/numeric.glsl>

#define nbl_glsl_PI 3.14159265359
#define nbl_glsl_RECIPROCAL_PI 0.318309886183
#define nbl_glsl_SQRT_RECIPROCAL_PI 0.56418958354

#define nbl_glsl_FLT_INF float(1.0/0.0)

#ifndef nbl_glsl_FLT_NAN
#define nbl_glsl_FLT_NAN uintBitsToFloat(0xFFffFFffu)
#endif

#endif
