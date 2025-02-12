// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_RANDOM_PCG_HLSL_INCLUDED_
#define _NBL_BUILTIN_GLSL_RANDOM_PCG_HLSL_INCLUDED_

namespace nbl
{
namespace hlsl
{

struct PCGStateHolder
{
    void pcg32_state_advance()
    {
        state = state * 747796405u + 2891336453u;
    }

    uint32_t state;
};

struct PCG32
{
    static PCG32 construct(NBL_CONST_REF_ARG(uint32_t) initialState)
    {
        PCGStateHolder stateHolder = {initialState};
        return PCG32(stateHolder);
    }

    uint32_t operator()()
    {
        const uint32_t word = ((stateHolder.state >> ((stateHolder.state >> 28u) + 4u)) ^ stateHolder.state) * 277803737u;
        const uint32_t result = (word >> 22u) ^ word;
        stateHolder.pcg32_state_advance();

        return result;
    }

    PCGStateHolder stateHolder;
};

struct PCG32x2
{
    static PCG32x2 construct(NBL_CONST_REF_ARG(uint32_t) initialState)
    {
        PCG32 rng = PCG32::construct(initialState);
        return PCG32x2(rng);
    }

    uint32_t2 operator()()
    {
        return uint32_t2(rng(), rng());
    }

    PCG rng;
};

}
}

#endif
