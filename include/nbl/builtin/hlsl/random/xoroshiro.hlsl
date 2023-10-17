
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_RANDOM_XOROSHIRO_HLSL_INCLUDED_
#define _NBL_BUILTIN_GLSL_RANDOM_XOROSHIRO_HLSL_INCLUDED_

<<<<<<< HEAD
#include <nbl/builtin/hlsl/math/functions.hlsl>

=======
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>
>>>>>>> 6fd2e3df3e4db9ef00e8a92fc4514ce3f56d77a7

namespace nbl
{
namespace hlsl
<<<<<<< HEAD
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
=======
{

struct Xoroshiro64StateHolder
{
	void xoroshiro64_state_advance()
	{
		state[1] ^= state[0];
		state[0] = rotl(state[0], 26u) ^ state[1] ^ (state[1]<<9u); // a, b
		state[1] = rotl(state[1], 13u); // c
	}
	
	uint32_t2 state;
};

struct Xoroshiro64Star
{
	static Xoroshiro64Star construct(NBL_CONST_REF_ARG(uint32_t2) initialState)
	{
		Xoroshiro64StateHolder stateHolder = {initialState};
		return Xoroshiro64Star(stateHolder);
	}
	
	uint32_t operator()()
	{
		const uint32_t result = stateHolder.state[0]*0x9E3779BBu;
		stateHolder.xoroshiro64_state_advance();

		return result;
	}
	
	Xoroshiro64StateHolder stateHolder;
};

struct Xoroshiro64StarStar
{
	static Xoroshiro64StarStar construct(NBL_CONST_REF_ARG(uint32_t2) initialState)
	{
		Xoroshiro64StateHolder stateHolder = {initialState};
		return Xoroshiro64StarStar(stateHolder);
	}
	
	uint32_t operator()()
	{
		const uint32_t result = rotl(stateHolder.state[0]*0x9E3779BBu,5u)*5u;
	    stateHolder.xoroshiro64_state_advance();
	
		return result;
	}

	Xoroshiro64StateHolder stateHolder;
};
>>>>>>> 6fd2e3df3e4db9ef00e8a92fc4514ce3f56d77a7

}
}

#endif