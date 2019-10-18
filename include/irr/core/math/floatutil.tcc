// Copyright (C) 2019 DevSH Graphics Programming Sp. z O.O.
// This file is part of the "IrrlichtBaW".
// For conditions of distribution and use, see LICENSE.md

#ifndef __IRR_FLOAT_UTIL_TCC_INCLUDED__
#define __IRR_FLOAT_UTIL_TCC_INCLUDED__


#include "irr/core/math/floatutil.h"
#include "matrix4SIMD.h"

namespace irr
{
namespace core
{

template<>
IRR_FORCE_INLINE float ROUNDING_ERROR<float>()
{
	return 0.000001f;
}
template<>
IRR_FORCE_INLINE double ROUNDING_ERROR<double>()
{
	return 0.00000001;
}
template<>
IRR_FORCE_INLINE vectorSIMDf ROUNDING_ERROR<vectorSIMDf>()
{
	return vectorSIMDf(ROUNDING_ERROR<float>());
}
template<>
IRR_FORCE_INLINE matrix3x4SIMD ROUNDING_ERROR<matrix3x4SIMD>()
{
	return matrix3x4SIMD(ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>());
}
template<>
IRR_FORCE_INLINE matrix4SIMD ROUNDING_ERROR<matrix4SIMD>()
{
	return matrix4SIMD(ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>());
}
template<typename T>
IRR_FORCE_INLINE T ROUNDING_ERROR()
{
	return T(0);
}


}
}

#endif
