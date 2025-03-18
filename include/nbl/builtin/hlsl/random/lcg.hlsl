// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_RANDOM_LCG_HLSL_INCLUDED_
#define _NBL_BUILTIN_GLSL_RANDOM_LCG_HLSL_INCLUDED_

namespace nbl
{
namespace hlsl
{

struct Lcg
{
    static Lcg construct(NBL_CONST_REF_ARG(uint32_t) state)
	{
        return Lcg(state);
    }
	
	uint32_t2 operator()()
	{
        uint32_t LCG_A = 1664525u;
        uint32_t LCG_C = 1013904223u;
        state = (LCG_A * state + LCG_C);
        state &= 0x00FFFFFF;
        return state;
    }

    uint32_t state;
};

}
}
#endif
