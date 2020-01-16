#ifndef _IRR_CORE_OWEN_SAMPLER_H_
#define _IRR_CORE_OWEN_SAMPLER_H_

#include <random>

#include "irr/core/Types.h"
#include "irr/core/sampling/SobolSampler.h"

namespace irr
{
namespace core
{


class OwenSampler : protected SobolSampler
{
	public:
		OwenSampler(uint32_t _dimensions, uint32_t _seed) : SobolSampler(_dimensions)
		{
			mersenneTwister.seed(_seed);

			constexpr uint32_t expectedMaxSamples = 4096u;
			flipBit.reserve(dimension*(expectedMaxSamples-1u)*SobolSampler::SOBOL_BITS);
		}
		~OwenSampler()
		{
		}
		
		// 
		inline uint32_t sample(uint32_t dim, uint32_t sampleNum)
		{
			uint32_t oldsample = SobolSampler::sample(dim,sampleNum);

			uint32_t retval = oldsample;
			for (uint32_t i=0; i<SobolSampler::SOBOL_BITS; i++)
			{
				uint32_t treeAboveSize = (0x1u<<i)-1u;
				uint32_t thisLevelIx = (oldsample>>(SobolSampler::SOBOL_BITS-i));

				uint64_t globalIndex = 0xffffffffull;
				globalIndex *= dim;
				globalIndex += treeAboveSize;
				globalIndex += thisLevelIx;

				bool flip;
				auto found = flipBit.find(globalIndex);
				if (found != flipBit.end())
					flip = found->second;
				else
				{
					flip = mersenneTwister()&0x80000000u;
					flipBit.insert({globalIndex,flip});
				}

				if (flip)
					retval ^= 0x1u<<(SobolSampler::SOBOL_BITS-1u-i);
			}

			return retval;
		}

	protected:
		std::mt19937 mersenneTwister;
		core::unordered_map<uint64_t,bool> flipBit;
};


}
}

#endif // _IRR_CORE_OWEN_SAMPLER_H_