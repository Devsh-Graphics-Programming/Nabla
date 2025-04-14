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
    using seed_type = uint32_t;

    static PCG32 construct(NBL_CONST_REF_ARG(seed_type) initialState)
    {
        PCG32 retval;
        retval.state = initialState;
        return retval;
    }

    uint32_t operator()()
    {
        const seed_type oldState = state;
        state = state * 747796405u + 2891336453u;
        const uint32_t word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
        const uint32_t result = (word >> 22u) ^ word;

        return result;
    }

    seed_type state;
};

}
}
}
#endif
