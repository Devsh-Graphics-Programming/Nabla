// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_RANDOM_SAMPLER_H_
#define __NBL_CORE_RANDOM_SAMPLER_H_

#include <random>

#include "irr/core/Types.h"

namespace irr
{
namespace core
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

	protected:
		std::mt19937 mersenneTwister;
	};


}
}

#endif // _IRR_CORE_RANDOM_SAMPLER_H_