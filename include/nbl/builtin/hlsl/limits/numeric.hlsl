
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

	
float float_epsilon(float n)
{
	return ieee754::fast_mul_exp2(n,-24);
}
float float_epsilon(int n)
{
	return float_epsilon(float(n));
}
float float_epsilon()
{
	return FLT_EPSILON;
}


}
}


#endif

