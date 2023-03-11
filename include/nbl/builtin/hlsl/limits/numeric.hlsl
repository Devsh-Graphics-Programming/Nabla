
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_LIMITS_NUMERIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_LIMITS_NUMERIC_INCLUDED_

#ifndef INT_MIN
#define INT_MIN -2147483648
#endif
#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

#ifndef UINT_MIN
#define UINT_MIN 0u
#endif
#ifndef UINT_MAX
#define UINT_MAX 4294967295u
#endif

#ifndef FLT_MIN
#define FLT_MIN 1.175494351e-38
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38
#endif

#ifndef FLT_INF
#define FLT_INF (1.f/0.f)
#endif

#ifndef FLT_NAN
#define FLT_NAN asfloat(0xFFffFFffu)
#endif

#ifndef FLT_EPSILON
#define	FLT_EPSILON 5.96046447754e-08
#endif 

#include <nbl/builtin/hlsl/ieee754.hlsl>


namespace nbl
{
namespace hlsl
{

template <typename T>
struct numeric_limits;

template <>
struct numeric_limits <float>
{
	static float nan() { return (0.0f / 0.0f); }
	static float min() { return 1.175494351e-38f; }
	static float max() { return 3.402823466e+38f; }
	static float epsilon() { return ieee754::float_epsilon(); }
	static float inf() { return 1.0f / 0.0f; }
};

// TODO int, uint specializations

}
}


#endif

