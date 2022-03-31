#ifndef _NBL_GLSL_BLIT_R11FG11FB10F_INCLUDED_
#define _NBL_GLSL_BLIT_R11FG11FB10F_INCLUDED_

#ifndef _NBL_GLSL_BLIT_DIM_COUNT_
	#error _NBL_GLSL_BLIT_DIM_COUNT_ must be defined
#endif

#ifndef _NBL_GLSL_BLIT_OUT_DESCRIPTOR_DEFINED_
	#error _NBL_GLSL_BLIT_OUT_DESCRIPTOR_DEFINED_ must be defined
#endif

uint to11bitFloat(in float _f32)
{
	const uint f32 = floatBitsToUint(_f32);

	if ((f32 & 0x80000000u) != 0u)
		return 0u;

	const uint f11MantissaMask = 0x3fu;
	const uint f11ExpMask = 0x1fu << 6;

	const int exp = int(((f32 >> 23) & 0xffu) - 127);
	const uint mantissa = f32 & 0x7fffffu;

	uint f11 = 0u;
	if (exp == 128) // inf / NaN
	{
		f11 = f11ExpMask;
		if (mantissa != 0u)
			f11 |= (mantissa & f11MantissaMask);
	}
	else if (exp > 15) // overflow converts to infinity
		f11 = f11ExpMask;
	else if (exp > -15)
	{
		const int e = exp + 15;
		const uint m = mantissa >> (23 - 6);
		f11 = (e << 6) | m;
	}

	return f11;
}

uint to10bitFloat(in float _f32)
{
	const uint f32 = floatBitsToUint(_f32);

	if ((f32 & 0x80000000u) != 0u) // negative numbers converts to 0 (represented by all zeroes in 10bit format)
		return 0;

	const uint f10MantissaMask = 0x1fu;
	const uint f10ExpMask = 0x1fu << 5;

	const int exp = int(((f32 >> 23) & 0xffu) - 127);
	const uint mantissa = f32 & 0x7fffffu;

	uint f10 = 0u;
	if (exp == 128) // inf / NaN
	{
		f10 = f10ExpMask;
		if (mantissa != 0u)
			f10 |= (mantissa & f10MantissaMask);
	}
	else if (exp > 15) // overflow, converts to infinity
		f10 = f10ExpMask;
	else if (exp > -15)
	{
		const int e = exp + 15;
		const uint m = mantissa >> (23 - 5);
		f10 = (e << 5) | m;
	}

	return f10;
}

void nbl_glsl_blit_setData(in nbl_glsl_blit_pixel_t value, in ivec3 coord)
{
#if NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 1)
	#define COORD coord.x
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 2)
	#define COORD coord.xy
#elif NBL_GLSL_EQUAL(_NBL_GLSL_BLIT_DIM_COUNT_, 3)
	#define COORD coord
#else
	#error _NBL_GLSL_BLIT_DIM_COUNT_ not supported
#endif

	const uint x = to11bitFloat(value.data.r);
	const uint y = to11bitFloat(value.data.g) << 11;
	const uint z = to10bitFloat(value.data.b) << 22;
	const uint encoded = z | y | x;

	imageStore(_NBL_GLSL_BLIT_OUT_DESCRIPTOR_DEFINED_, COORD, uvec4(encoded, 0, 0, 0));

#undef COORD
}

#endif