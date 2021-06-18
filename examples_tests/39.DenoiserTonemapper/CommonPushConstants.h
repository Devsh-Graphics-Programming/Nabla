// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifdef __cplusplus
	#define int int32_t
	#define uint uint32_t
	struct vec2 {float x,y;};
	#define mat3 nbl::core::matrix3x4SIMD
#endif
struct CommonPushConstants
{
	uint inImageTexelPitch[3];
	uint imageWidth;
	uint imageHeight;
	uint padding;
	vec2 kernel_half_pixel_size;
	
	// luma meter and tonemapping var but also for denoiser
	uint percentileRange[2];
	uint intensityBufferDWORDOffset;
	float denoiserExposureBias;

	uint flags;
	// for the tonemapper
	uint tonemappingOperator;
	float tonemapperParams[2];


	mat3 normalMatrix;
};
#ifdef __cplusplus
	#undef int
	#undef uint
	#undef mat3
#endif