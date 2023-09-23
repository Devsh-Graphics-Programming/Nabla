
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_RANDOM_XOROSHIRO_HLSL_INCLUDED_
#define _NBL_BUILTIN_GLSL_RANDOM_XOROSHIRO_HLSL_INCLUDED_

//#include <nbl/builtin/hlsl/math/functions.hlsl>

// TODO[Przemek]: include functions.hlsl instead
uint32_t rotl(NBL_CONST_REF_ARG(uint32_t) x, NBL_CONST_REF_ARG(uint32_t) k)
{
   return (x<<k) | (x>>(32u-k));
}

namespace nbl
{
namespace hlsl
{

typedef uint2 xoroshiro64star_state_t;
typedef uint2 xoroshiro64starstar_state_t;

namespace impl
{
	uint2 xoroshiro64_state_advance(uint2 state)
	{
		state[1] ^= state[0];
		state[0] = rotl(state[0], 26u) ^ state[1] ^ (state[1]<<9u); // a, b
		state[1] = rotl(state[1], 13u); // c
		
		return state;
	}
}

struct Xoroshriro64Star
{
	static Xoroshriro64Star construct(xoroshiro64star_state_t initialState)
	{
		return { initialState };
	}
	
	uint32_t operator()()
	{
		const uint32_t result = state[0]*0x9E3779BBu;
		state = impl::xoroshiro64_state_advance(state);

		return result;
	}

	xoroshiro64star_state_t state;
};

struct Xoroshriro64StarStar
{
	static Xoroshriro64StarStar construct(xoroshiro64starstar_state_t initialState)
	{
		return { initialState };
	}
	
	uint32_t operator()()
	{
		const uint32_t result = rotl(state[0]*0x9E3779BBu,5u)*5u;
	    state = impl::xoroshiro64_state_advance(state);
	
		return result;
	}

	xoroshiro64starstar_state_t state;
};

}
}

#endif