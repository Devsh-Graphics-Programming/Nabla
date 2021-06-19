// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_LIMITS_NUMERIC_INCLUDED_
#define _NBL_LIMITS_NUMERIC_INCLUDED_

#ifndef nbl_glsl_INT_MIN
#define nbl_glsl_INT_MIN -2147483648
#endif
#ifndef nbl_glsl_INT_MAX
#define nbl_glsl_INT_MAX 2147483647
#endif

#ifndef nbl_glsl_UINT_MIN
#define nbl_glsl_UINT_MIN 0u
#endif
#ifndef nbl_glsl_UINT_MAX
#define nbl_glsl_UINT_MAX 4294967295u
#endif

#ifndef nbl_glsl_FLT_MIN
#define nbl_glsl_FLT_MIN 1.175494351e-38
#endif

#ifndef nbl_glsl_FLT_MAX
#define nbl_glsl_FLT_MAX 3.402823466e+38
#endif

#ifndef nbl_glsl_FLT_INF
#define nbl_glsl_FLT_INF (1.f/0.f)
#endif

#ifndef nbl_glsl_FLT_NAN
#define nbl_glsl_FLT_NAN uintBitsToFloat(0xFFffFFffu)
#endif

#ifndef nbl_glsl_FLT_EPSILON
#define	nbl_glsl_FLT_EPSILON 5.96046447754e-08
#endif 


// TODO: move to some other header (maybe ieee754.glsl)
uint nbl_glsl_ieee754_extract_biased_exponent(float x)
{
	return bitfieldExtract(floatBitsToUint(x),23,8);
}
float nbl_glsl_ieee754_replace_biased_exponent(float x, uint exp_plus_bias)
{
	return uintBitsToFloat(bitfieldInsert(floatBitsToUint(x),exp_plus_bias,23,8));
}
// performs no overflow tests, returns x*exp2(n)
float nbl_glsl_ieee754_fast_mul_exp2(float x, int n)
{
	return nbl_glsl_ieee754_replace_biased_exponent(x,nbl_glsl_ieee754_extract_biased_exponent(x)+uint(n));
}


float nbl_glsl_numeric_limits_float_epsilon(float n)
{
	return nbl_glsl_ieee754_fast_mul_exp2(n,-24);
}
float nbl_glsl_numeric_limits_float_epsilon(int n)
{
	return nbl_glsl_numeric_limits_float_epsilon(float(n));
}
float nbl_glsl_numeric_limits_float_epsilon()
{
	return nbl_glsl_FLT_EPSILON;
}

float nbl_glsl_ieee754_gamma(float n)
{
	const float a = nbl_glsl_numeric_limits_float_epsilon(n);
	return a/(1.f-a);
}
float nbl_glsl_ieee754_gamma(uint n)
{
	return nbl_glsl_ieee754_gamma(float(n));
}

#endif
