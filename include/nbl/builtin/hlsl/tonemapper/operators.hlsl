// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_TONE_MAPPER_OPERATORS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TONE_MAPPER_OPERATORS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{

struct ReinhardParams
{
	float32_t keyAndManualLinearExposure;
	float32_t rcpWhite2;
};

struct ACESParams
{
	float32_t gamma; // 1.0
	float32_t exposure; // actualExposure+midGrayLog2
};


float32_t3 reinhard(ReinhardParams params, float32_t3 rawCIEXYZcolor)
{
	float32_t exposureFactors = params.keyAndManualLinearExposure;
	float32_t exposedLuma = rawCIEXYZcolor.y * exposureFactors;
	float32_t colorMultiplier = (exposureFactors * (1.0 + exposedLuma * params.rcpWhite2) / (1.0 + exposedLuma));
	return rawCIEXYZcolor * colorMultiplier;
}

float32_t3 aces(ACESParams params, float32_t3 rawCIEXYZcolor)
{
	float32_t3 tonemapped = rawCIEXYZcolor;
	if (tonemapped.y > 1.175494351e-38)
		tonemapped *= exp2(log2(tonemapped.y) * (params.gamma - 1.0) + (params.exposure) * params.gamma);

	// XYZ => RRT_SAT
	// this seems to be a matrix for some hybrid colorspace, coefficients are somewhere inbetween BT2020 and ACEScc(t)
	const float32_t3x3 XYZ_RRT_Input = float32_t3x3(
		float32_t3(1.594168310, -0.262608051, -0.231993079),
		float32_t3(-0.6332771780, 1.5840380200, 0.0164147373),
		float32_t3(0.00892840419, 0.03648501260, 0.87711471300)
	);

	// this is obviously fitted to some particular simulated sensor/film and display
	float32_t3 v = mul(XYZ_RRT_Input, tonemapped);
	float32_t3 a = v * (v + float32_t3(0.0245786)) - float32_t3(0.000090537);
	float32_t3 b = v * (v * float32_t(0.983729) + float32_t3(0.4329510)) + float32_t3(0.238081);
	v = a / b;

	// ODT_SAT => XYZ
	// this seems to be a matrix for some hybrid colorspace, coefficients are similar to AdobeRGB,BT2020 and ACEScc(t)
	const float32_t3x3 ODT_XYZ_Output = float32_t3x3(
		float32_t3(0.624798000, 0.164064825, 0.161605373),
		float32_t3(0.268048108, 0.674283803, 0.057667464),
		float32_t3(0.0157514643, 0.0526682511, 1.0204007600)
	);
	return mul(ODT_XYZ_Output, v);
}

// ideas for more operators https://web.archive.org/web/20191226154550/http://cs.columbia.edu/CAVE/software/softlib/dorf.php
// or get proper ACES RRT and ODTs
// https://partnerhelp.netflixstudios.com/hc/en-us/articles/360000622487-I-m-using-ACES-Which-Output-Transform-should-I-use-

}
}

#endif