// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_TONE_MAPPER_OPERATORS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TONE_MAPPER_OPERATORS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace tonemapper
{

template<typename T = float32_t>
struct Reinhard
{
	using float_t = enable_if_t<is_floating_point<T>::value, T>;
	using float_t3 = typename conditional<is_same_v<float_t, float32_t>, float32_t3, float16_t3>::type;
	using this_t = Reinhard<float_t>;
	static this_t create(float_t EV, float_t key = 0.18f, float_t WhitePointRelToEV = 16.f)
	{
		this_t retval;
		retval.keyAndManualLinearExposure = key * exp2(EV);
		retval.rcpWhite2 = 1.f / (WhitePointRelToEV * WhitePointRelToEV);
		return retval;
	}

	float_t3 operator()(float_t3 rawCIEXYZcolor) {
		float_t exposureFactors = keyAndManualLinearExposure;
		float_t exposedLuma = rawCIEXYZcolor.y * exposureFactors;
		float_t colorMultiplier = (exposureFactors * (1.0 + exposedLuma * rcpWhite2) / (1.0 + exposedLuma));
		return rawCIEXYZcolor * colorMultiplier;
	}

	float_t keyAndManualLinearExposure;
	float_t rcpWhite2;
};

template<typename T = float32_t>
struct ACES
{
	using float_t = enable_if_t<is_floating_point<T>::value, T>;
	using float_t3 = typename conditional<is_same_v<float_t, float32_t>, float32_t3, float16_t3>::type;
	using float_t3x3 = typename conditional<is_same_v<float_t, float32_t>, float32_t3x3, float16_t3x3>::type;

	using this_t = ACES<T>;
	static this_t create(float_t EV, float_t key = 0.18f, float_t Contrast = 1.f) {
		this_t retval;
		retval.gamma = Contrast;
		const float_t reinhardMatchCorrection = 0.77321666f; // middle grays get exposed to different values between tonemappers given the same key
		retval.exposure = EV + log2(key * reinhardMatchCorrection);
		return retval;
	}

	float_t3 operator()(float_t3 rawCIEXYZcolor) {
		float_t3 tonemapped = rawCIEXYZcolor;
		if (tonemapped.y > 1.175494351e-38)
			tonemapped *= exp2(log2(tonemapped.y) * (gamma - 1.0) + (exposure) * gamma);

		// XYZ => RRT_SAT
		// this seems to be a matrix for some hybrid colorspace, coefficients are somewhere inbetween BT2020 and ACEScc(t)
		const float_t3x3 XYZ_RRT_Input = float_t3x3(
			float_t3(1.594168310, -0.262608051, -0.231993079),
			float_t3(-0.6332771780, 1.5840380200, 0.0164147373),
			float_t3(0.00892840419, 0.03648501260, 0.87711471300)
		);

		// this is obviously fitted to some particular simulated sensor/film and display
		float_t3 v = mul(XYZ_RRT_Input, tonemapped);
		float_t3 a = v * (v + promote<float_t3>(0.0245786)) - promote<float_t3>(0.000090537);
		float_t3 b = v * (v * promote<float_t3>(0.983729) + promote<float_t3>(0.4329510)) + promote<float_t3>(0.238081);
		v = a / b;

		// ODT_SAT => XYZ
		// this seems to be a matrix for some hybrid colorspace, coefficients are similar to AdobeRGB,BT2020 and ACEScc(t)
		const float_t3x3 ODT_XYZ_Output = float_t3x3(
			float_t3(0.624798000, 0.164064825, 0.161605373),
			float_t3(0.268048108, 0.674283803, 0.057667464),
			float_t3(0.0157514643, 0.0526682511, 1.0204007600)
		);
		return mul(ODT_XYZ_Output, v);
	}

	float_t gamma; // 1.0
	float_t exposure; // actualExposure+midGrayLog2
};

// ideas for more operators https://web.archive.org/web/20191226154550/http://cs.columbia.edu/CAVE/software/softlib/dorf.php
// or get proper ACES RRT and ODTs
// https://partnerhelp.netflixstudios.com/hc/en-us/articles/360000622487-I-m-using-ACES-Which-Output-Transform-should-I-use-

}
}
}

#endif