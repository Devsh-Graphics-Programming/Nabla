// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_RANDOM_LCG_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RANDOM_LCG_HLSL_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace random
{

struct Lcg
{
    using seed_type = uint32_t;

    static Lcg create(NBL_CONST_REF_ARG(seed_type) _state)
	{
        Lcg retval;
        retval.state = _state;
        return retval;
    }

	uint32_t operator()()
	{
        uint32_t LCG_A = 1664525u;
        uint32_t LCG_C = 1013904223u;
        state = (LCG_A * state + LCG_C);
        // is this mask supposed to be here? because usually we turn random uints to float by dividing by UINT32_MAX
        state &= 0x00FFFFFF;
        return state;
    }

    seed_type state;
};

}
}
}
#endif
