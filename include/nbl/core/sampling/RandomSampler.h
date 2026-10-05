// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_RANDOM_SAMPLER_H_
#define __NBL_CORE_RANDOM_SAMPLER_H_

#include <random>
#include "nbl/core/decl/Types.h"

namespace nbl::core
{

	class RandomSampler
	{
	public:
		RandomSampler(uint32_t _seed)
		{
			mersenneTwister.seed(_seed);
		}

		// 
		inline uint32_t nextSample()
		{
			return mersenneTwister();
		}

		// Returns a float in [0, 1)
		inline float nextFloat()
		{
			// 1 / 2^32
			constexpr float norm = 1.0f / 4294967296.0f;
			return mersenneTwister() * norm;
		}

		// Returns a float in [min, max)
		inline float nextFloat(float min, float max)
		{
			constexpr float norm = 1.0f / 4294967296.0f;
			return min + (mersenneTwister() * norm) * (max - min);
		}

	protected:
		std::mt19937 mersenneTwister;
	};


}

#endif