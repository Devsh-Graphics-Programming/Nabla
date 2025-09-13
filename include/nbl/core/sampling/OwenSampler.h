// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_CORE_OWEN_SAMPLER_H_
#define __NBL_CORE_CORE_OWEN_SAMPLER_H_

#include "nbl/core/sampling/RandomSampler.h"
#include "nbl/core/sampling/SobolSampler.h"

namespace nbl
{
namespace core
{

	//! TODO: make the tree sampler/generator configurable and let RandomSampler be default
	template<class SequenceSampler=SobolSampler>
	class OwenSampler : protected SequenceSampler
	{
	public:
		OwenSampler(uint32_t _dimensions, uint32_t _seed) : SequenceSampler(_dimensions)
		{
			mersenneTwister.seed(_seed);
			cachedFlip.resize(MAX_SAMPLES-1u);
			resetDimensionCounter(0u);
		}
		~OwenSampler()
		{
		}

		// 
		inline uint32_t sample(uint32_t dim, uint32_t sampleNum)
		{
			if (dim>lastDim)
				resetDimensionCounter(dim);
			else if (dim<lastDim)
				assert(false);

			uint32_t oldsample = SequenceSampler::sample(dim,sampleNum);
			#ifdef _NBL_DEBUG
				assert(sampleNum<MAX_SAMPLES);
				if (sampleNum)
					assert((oldsample&(0x7fffffffu>>hlsl::findMSB(sampleNum))) == 0u);
				else
					assert(oldsample == 0u);
			#endif
			constexpr uint32_t lastLevelStart = MAX_SAMPLES/2u-1u;
			uint32_t index = oldsample>>(OUT_BITS+1u - MAX_SAMPLES_LOG2);
			index += lastLevelStart;

			return oldsample^cachedFlip[index];
		}

		//!
		inline void resetDimensionCounter(uint32_t dimension)
		{
			/** NOTES:
			- For 64k samples, we can store their positions in uint16_t
			- The last leves of Owen Tree can be collapsed to a single node (because trailing bits are always 00000.....)
			- The above can be stored in 1x array of sample count uint16_t/uint32_t per Dimension
			- We should store samples as uint32_t always because the total amount of memory to fetch is always the same
			**/
			for (uint32_t i=0u; i<MAX_SAMPLES-1u; i++) 
			{
				uint32_t randMask = (i<(MAX_SAMPLES/2u-1u)) ? 0x80000000u:0xffffffffu;
				cachedFlip[i] = mersenneTwister()&(randMask>>getTreeDepth(i));
			}
			for (uint32_t i=1u; i<MAX_SAMPLES_LOG2; i++)
			{
				uint32_t previousLevelStart = (0x1u<<(i-1u))-1u;
				uint32_t currentLevelStart = (0x1u<<i)-1u;
				uint32_t currentLevelSize = 0x1u<<i;
				for (uint32_t j=0u; j<currentLevelSize; j++)
					cachedFlip[currentLevelStart+j] |= cachedFlip[previousLevelStart+(j>>1u)];
				#ifdef _NBL_DEBUG
				for (uint32_t j=0u; j<currentLevelSize; j+=2)
				{
					const uint32_t highBitMask = 0xffffffffu<<(OUT_BITS-i);
					uint32_t left = cachedFlip[currentLevelStart+j];
					uint32_t right = cachedFlip[currentLevelStart+j+1];
					assert(((left^right)&highBitMask)==0u);
					assert((left&right&highBitMask)==cachedFlip[previousLevelStart+(j>>1u)]);
				}
				#endif
			}
			lastDim = dimension;
		}

	protected:
		// if we don't limit the sample count, then due to IEEE754 precision, we'll get duplicate sample coordinate values, ruining the net property
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t OUT_BITS = sizeof(uint32_t)*8u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MAX_SAMPLES_LOG2 = 24u;
		_NBL_STATIC_INLINE_CONSTEXPR uint32_t MAX_SAMPLES = 0x1u<<MAX_SAMPLES_LOG2;

		inline uint32_t getTreeDepth(uint32_t sampleNum)
		{
			return hlsl::findMSB(sampleNum+1u);
		}

		std::mt19937 mersenneTwister;
		uint32_t lastDim;
		core::vector<uint32_t> cachedFlip;
	};


}
}

#endif
