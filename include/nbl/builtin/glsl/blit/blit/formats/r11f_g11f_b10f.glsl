#ifndef _NBL_GLSL_BLIT_R11FG11FB10F_INCLUDED_
#define _NBL_GLSL_BLIT_R11FG11FB10F_INCLUDED_

uint nbl_glsl_encode_ufloat(in float _f32, in uint mantissaMask, in uint mantissaBits)
{
	const uint f32 = floatBitsToUint(_f32);

	if ((f32 & 0x80000000u) != 0u) // negative numbers converts to 0 (represented by all zeroes)
		return 0;

	const uint expMask = 0x1fu << mantissaBits;

	const int exp = int(((f32 >> 23) & 0xffu) - 127);
	const uint mantissa = f32 & 0x7fffffu;

	uint f = 0u;
	if (exp == 128) // inf / NaN
	{
		f = expMask;
		if (mantissa != 0u)
			f |= (mantissa & mantissaMask);
	}
	else if (exp > 15) // overflow, converts to infinity
		f = expMask;
	else if (exp > -15)
	{
		const int e = exp + 15;
		const uint m = mantissa >> (23 - mantissaBits);
		f = (e << mantissaBits) | m;
	}

	return f;
}

uint to11bitFloat(in float _f32)
{
	const uint mantissaMask = 0x3fu;
	const uint mantissaBits = 6;
	return nbl_glsl_encode_ufloat(_f32, mantissaMask, mantissaBits);
}

uint to10bitFloat(in float _f32)
{
	const uint mantissaMask = 0x1fu;
	const uint mantissaBits = 5;
	return nbl_glsl_encode_ufloat(_f32, mantissaMask, mantissaBits);
}

uvec4 nbl_glsl_blit_formats_encode(in vec4 value)
{
	const uint x = to11bitFloat(value.r);
	const uint y = to11bitFloat(value.g) << 11;
	const uint z = to10bitFloat(value.b) << 22;
	const uint encoded = z | y | x;

	return uvec4(encoded, 0, 0, 0);
}

#endif