// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_LUMA_METER_COMMON_INCLUDED_
#define _NBL_BUILTIN_HLSL_LUMA_METER_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{
namespace luma_meter
{

struct MeteringWindow
{
	using this_t = MeteringWindow;
	float32_t2 meteringWindowScale;
	float32_t2 meteringWindowOffset;

	static this_t create(float32_t2 scale, float32_t2 offset) {
		this_t retval;
		retval.meteringWindowScale = scale;
		retval.meteringWindowOffset = offset;
		return retval;
	}
};

}
}
}

#endif