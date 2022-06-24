#ifndef _NBL_BUILTIN_GLSL_IEE754_H_INCLUDED_
#define _NBL_BUILTIN_GLSL_IEE754_H_INCLUDED_

uint nbl_glsl_ieee754_extract_biased_exponent(float x)
{
	return bitfieldExtract(floatBitsToUint(x), 23, 8);
}
float nbl_glsl_ieee754_replace_biased_exponent(float x, uint exp_plus_bias)
{
	return uintBitsToFloat(bitfieldInsert(floatBitsToUint(x), exp_plus_bias, 23, 8));
}
// performs no overflow tests, returns x*exp2(n)
float nbl_glsl_ieee754_fast_mul_exp2(float x, int n)
{
	return nbl_glsl_ieee754_replace_biased_exponent(x, nbl_glsl_ieee754_extract_biased_exponent(x) + uint(n));
}
uint nbl_glsl_ieee754_compute_mantissa_mask(in uint mantissaBits)
{
	return (0x1u << mantissaBits) - 1;
}
uint nbl_glsl_ieee754_extract_mantissa(in float x)
{
	return (floatBitsToUint(x) & 0x7fffffu);
}

#endif