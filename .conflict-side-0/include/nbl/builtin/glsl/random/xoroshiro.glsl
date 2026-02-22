// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_RANDOM_XOROSHIRO_GLSL_INCLUDED_
#define _NBL_BUILTIN_GLSL_RANDOM_XOROSHIRO_GLSL_INCLUDED_

#include <nbl/builtin/glsl/math/functions.glsl>

#define nbl_glsl_xoroshiro64star_state_t uvec2
#define nbl_glsl_xoroshiro64starstar_state_t uvec2
void nbl_glsl_xoroshiro64_state_advance(inout uvec2 state)
{
	state[1] ^= state[0];
	state[0] = nbl_glsl_rotl(state[0], 26u) ^ state[1] ^ (state[1]<<9u); // a, b
	state[1] = nbl_glsl_rotl(state[1], 13u); // c
}

uint nbl_glsl_xoroshiro64star(inout nbl_glsl_xoroshiro64starstar_state_t state)
{
	const uint result = state[0]*0x9E3779BBu;

    nbl_glsl_xoroshiro64_state_advance(state);

	return result;
}
uint nbl_glsl_xoroshiro64starstar(inout nbl_glsl_xoroshiro64starstar_state_t state)
{
	const uint result = nbl_glsl_rotl(state[0]*0x9E3779BBu,5u)*5u;
    
    nbl_glsl_xoroshiro64_state_advance(state);

	return result;
}

#endif