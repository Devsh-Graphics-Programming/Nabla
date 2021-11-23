// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_MATRIX_UTIL_H_INCLUDED_
#define _NBL_MATRIX_UTIL_H_INCLUDED_

#include "matrix4SIMD.h"
#include "matrix3x4SIMD.h"

namespace nbl::core
{


//! TODO: OPTIMIZE THIS, DON'T PROMOTE THE MATRIX IF DON'T HAVE TO
inline matrix4SIMD concatenateBFollowedByA(const matrix4SIMD& _a, const matrix3x4SIMD& _b)
{
    return concatenateBFollowedByA(_a, matrix4SIMD(_b));
}
/*
inline matrix4SIMD concatenateBFollowedByAPrecisely(const matrix4SIMD& _a, const matrix3x4SIMD& _b)
{
    return concatenateBFollowedByAPrecisely(_a, matrix4SIMD(_b));
}
*/

}

#endif
