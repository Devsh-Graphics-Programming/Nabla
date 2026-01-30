// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_TONE_MAPPER_OPERATORS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TONE_MAPPER_OPERATORS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"

namespace nbl
{
namespace hlsl
{
namespace tonemapper
{

template<typename T NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeScalar<T>)
struct Reinhard
{
	using float_t = enable_if_t<is_floating_point<T>::value, T>;
	using float_t3 = vector<float_t, 3>;
	using this_t = Reinhard<float_t>;

	static this_t create(float_t EV, float_t key = 0.18f, float_t WhitePointRelToEV = 16.f)
	{
		this_t retval;

		const float_t unit = 1.0;
		retval.keyAndManualLinearExposure = key * exp2(-EV);
		retval.rcpWhite2 = unit / (WhitePointRelToEV * WhitePointRelToEV);

		return retval;
	}

	float_t3 operator()(float_t3 rawCIEXYZcolor)
    {
		const float_t unit = 1.0;
		float_t exposureFactors = keyAndManualLinearExposure;
		float_t exposedLuma = rawCIEXYZcolor.y * exposureFactors;
		float_t colorMultiplier = (exposureFactors * (unit + exposedLuma * rcpWhite2) / (unit + exposedLuma));
		return rawCIEXYZcolor * colorMultiplier;
	}

	float_t keyAndManualLinearExposure;
	float_t rcpWhite2;
};

}
}
}

#endif