// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_RANDOM_PCG_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RANDOM_PCG_HLSL_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace random
{

struct Pcg
{
	static Pcg create(const uint32_t initialState)
	{
		Pcg retval;
		retval.state = initialState;
		return retval;
	}
	
	uint32_t operator()()
	{
        const uint32_t tmp = state * 747796405u + 2891336453u;
        const uint32_t word = ((tmp >> ((tmp >> 28u) + 4u)) ^ tmp) * 277803737u;
        state = (word >> 22u) ^ word;
		return state;
    }
	
	uint32_t state;
};

}
}
}
#endif
