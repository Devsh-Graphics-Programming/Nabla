
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_IEE754_H_INCLUDED_
#define _NBL_BUILTIN_HLSL_IEE754_H_INCLUDED_

namespace nbl
{
namespace hlsl
{


float numeric_limits::float_epsilon(float n);
float numeric_limits::float_epsilon(int n);
float numeric_limits::float_epsilon();

uint bitfieldExtract(uint value, int offset, int bits)
{
	uint mask = uint((1 << bits) - 1);
	return uint(value >> offset) & mask;
}
uint bitfieldInsert(uint value, int insert, int offset, int bits)
{
	uint mask = ~(0xffffffff << bits) << offset;
	mask = ~mask;
	value &= mask;
	return value | (insert << offset);
}

namespace ieee754
{

uint exponent_bias(in uint exponentBits)
{
	return (0x1u << (exponentBits - 1)) - 1;
}
uint extract_biased_exponent(float x)
{
	return bitfieldExtract(asuint(x), 23, 8);
}
int extract_exponent(float x)
{
	return int(extract_biased_exponent(x) - exponent_bias(8));
}
uint compute_exponent_mask(in uint exponentBits, in uint mantissaBits)
{
	return ((1 << exponentBits) - 1) << mantissaBits;
}
float replace_biased_exponent(float x, uint exp_plus_bias)
{
	return asfloat(bitfieldInsert(asuint(x), exp_plus_bias, 23, 8));
}
// performs no overflow tests, returns x*exp2(n)
float fast_mul_exp2(float x, int n)
{
	return replace_biased_exponent(x, extract_biased_exponent(x) + uint(n));
}
uint compute_mantissa_mask(in uint mantissaBits)
{
	return (0x1u << mantissaBits) - 1;
}
uint extract_mantissa(in float x)
{
	return (asuint(x) & 0x7fffffu);
}
float true_min(in uint exponentBits, in uint mantissaBits)
{
	return exp2(1 - int(exponent_bias(exponentBits)) - mantissaBits);
}
float min(in uint exponentBits, in uint mantissaBits)
{
	const float e = exp2(1 - int(exponent_bias(exponentBits)));
	const uint m = 0x1u << (23 - mantissaBits);
	return asfloat(asuint(e) | m);
}
float max(in uint exponentBits, in uint mantissaBits)
{
	const uint biasedMaxExp = (((1 << exponentBits) - 1) - 1); // `(1 << exponentBits) - 1` is reserved for Inf/NaN.
	const float e = exp2(biasedMaxExp - int(exponent_bias(exponentBits)));
	const uint m = 0x7fFFffu & (0x7fFFffu << (23 - mantissaBits));
	return asfloat(asuint(e) | m);
}
uint encode_ufloat_impl(in int exponent, in uint exponentBits, in uint mantissa, in uint mantissaBits)
{
	const uint expBias = exponent_bias(exponentBits);
	const uint e = uint(exponent + expBias);
	const uint m = mantissa >> (23 - mantissaBits);
	const uint encodedValue = (e << mantissaBits) | m;
	return encodedValue;
}



float gamma(float n)
{
	const float a = numeric_limits::float_epsilon(n);
	return a / (1.f - a);
}
float rcpgamma(float n)
{
	const float a = numeric_limits::float_epsilon(n);
	return 1.f / a - 1.f;
}

float gamma(uint n)
{
	return gamma(float(n));
}
float rcpgamma(uint n)
{
	return rcpgamma(float(n));
}

float3 add_with_bounds_wo_gamma(out float3 error, in float3 a, in float3 a_error, in float3 b, in float3 b_error)
{
	error = (a_error + b_error) / numeric_limits::float_epsilon(1u);
	float3 sum = a + b;
	error += abs(sum);
	return sum;
}
float3 sub_with_bounds_wo_gamma(out float3 error, in float3 a, in float3 a_error, in float3 b, in float3 b_error)
{
	error = (a_error + b_error) / numeric_limits::float_epsilon(1u);
	float3 sum = a - b;
	error += abs(sum);
	return sum;
}
float3 mul_with_bounds_wo_gamma(out float3 error, in float3 a, in float3 a_error, in float b, in float b_error)
{
	float3 crossCorrelationA = abs(a) * b_error;
	float3 crossCorrelationB = a_error * abs(b);
	error = (crossCorrelationB + crossCorrelationA + crossCorrelationB * crossCorrelationA) / numeric_limits::float_epsilon(1u);
	float3 product = a * b;
	error += abs(product);
	return product;
}

}
}
}

#endif