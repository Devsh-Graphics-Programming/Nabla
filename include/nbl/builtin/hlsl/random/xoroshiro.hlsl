
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_RANDOM_XOROSHIRO_HLSL_INCLUDED_
#define _NBL_BUILTIN_GLSL_RANDOM_XOROSHIRO_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/math/functions.hlsl>


namespace nbl
{
namespace hlsl
{

#define xoroshiro64star_state_t uint2
#define xoroshiro64starstar_state_t uint2

void xoroshiro64_state_advance(inout uint2 state)
{
	state[1] ^= state[0];
	state[0] = math::rotl(state[0], 26u) ^ state[1] ^ (state[1]<<9u); // a, b
	state[1] = math::rotl(state[1], 13u); // c
}

uint xoroshiro64star(inout xoroshiro64starstar_state_t state)
{
	const uint result = state[0]*0x9E3779BBu;

    xoroshiro64_state_advance(state);

	return result;
}
uint xoroshiro64starstar(inout xoroshiro64starstar_state_t state)
{
	const uint result = math::rotl(state[0]*0x9E3779BBu,5u)*5u;
    
    xoroshiro64_state_advance(state);

	return result;
}

}
}

#endif