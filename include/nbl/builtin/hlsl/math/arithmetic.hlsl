// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_ARITHMETIC_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{
	//! WARNING: ONLY WORKS FOR `dividendMsb<=2^23` DUE TO FP32 ABUSE !!!
	uint32_t fastDivide_int23_t(in uint32_t dividendMsb, in uint32_t dividendLsb, in uint32_t divisor)
	{
		const float msbRatio = float(dividendMsb) / float(divisor);
		const uint32_t quotient = uint32_t((msbRatio * (~0u)) + msbRatio) + dividendLsb / divisor;
		return quotient;
	}
}
}
}

#endif
