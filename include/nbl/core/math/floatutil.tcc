// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_FLOAT_UTIL_TCC_INCLUDED__
#define __NBL_CORE_FLOAT_UTIL_TCC_INCLUDED__


#include "nbl/core/math/floatutil.h"
#include "matrix4SIMD.h"

namespace nbl
{
namespace core
{

template<>
NBL_FORCE_INLINE float ROUNDING_ERROR<float>()
{
	return 0.000001f;
}
template<>
NBL_FORCE_INLINE double ROUNDING_ERROR<double>()
{
	return 0.00000001;
}
template<>
NBL_FORCE_INLINE vectorSIMDf ROUNDING_ERROR<vectorSIMDf>()
{
	return vectorSIMDf(ROUNDING_ERROR<float>());
}
template<>
NBL_FORCE_INLINE matrix3x4SIMD ROUNDING_ERROR<matrix3x4SIMD>()
{
	return matrix3x4SIMD(ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>());
}
template<>
NBL_FORCE_INLINE matrix4SIMD ROUNDING_ERROR<matrix4SIMD>()
{
	return matrix4SIMD(ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>(),ROUNDING_ERROR<vectorSIMDf>());
}
template<typename T>
NBL_FORCE_INLINE T ROUNDING_ERROR()
{
	return T(0);
}


}
}

#endif
