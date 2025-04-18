// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_RANDOM_XOROSHIRO_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_RANDOM_XOROSHIRO_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

#include <nbl/builtin/hlsl/bit.hlsl>

namespace nbl
{
namespace hlsl
{
// TODO
//namespace random
//{

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
	using seed_type = uint32_t2;

	// TODO: create
	static Xoroshiro64Star construct(NBL_CONST_REF_ARG(seed_type) initialState)
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
	using seed_type = uint32_t2;

	// TODO: create
	static Xoroshiro64StarStar construct(NBL_CONST_REF_ARG(seed_type) initialState)
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

//}
}
}

#endif