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

struct PCG32
{
    static PCG32 construct(NBL_CONST_REF_ARG(uint32_t) initialState)
    {
        PCG32 retval;
        retval.state = initialState;
        return retval;
    }

    uint32_t operator()()
    {
        const uint32_t oldState = state;
        state = state * 747796405u + 2891336453u;
        const uint32_t word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
        const uint32_t result = (word >> 22u) ^ word;

        return result;
    }

    uint32_t state;
};

}
}
}
#endif
