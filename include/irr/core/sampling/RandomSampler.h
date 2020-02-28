#ifndef _IRR_CORE_RANDOM_SAMPLER_H_
#define _IRR_CORE_RANDOM_SAMPLER_H_

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