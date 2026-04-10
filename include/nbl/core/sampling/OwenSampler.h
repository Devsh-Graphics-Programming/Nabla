// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_CORE_CORE_OWEN_SAMPLER_H_
#define _NBL_CORE_CORE_OWEN_SAMPLER_H_

#include "nbl/core/sampling/RandomSampler.h"
#include "nbl/core/sampling/SobolSampler.h"

namespace nbl::core
{

//! TODO: make the tree sampler/generator configurable and let RandomSampler be default
template<class SequenceSampler=SobolSampler>
class OwenSampler final : protected SequenceSampler
{
		// if we don't limit the sample count, then due to IEEE754 precision, we'll get duplicate sample coordinate values, ruining the net property
		constexpr static inline uint32_t OUT_BITS = sizeof(uint32_t)*8u;
		constexpr static inline uint32_t MAX_SAMPLES_LOG2 = 24u;
		constexpr static inline uint32_t MAX_SAMPLES = 0x1u<<MAX_SAMPLES_LOG2;

	public:
		inline OwenSampler(uint32_t _dimensions, uint32_t _seed) : SequenceSampler(_dimensions), seed(_seed) {}
		inline ~OwenSampler() = default;

		struct SDimensionSampler final : public core::Unmovable
		{
				inline uint32_t sample(uint32_t sampleNum) const
				{
					const uint32_t oldsample = sampler.sample(dimension,sampleNum);
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

			private:
				friend class OwenSampler;
				inline SDimensionSampler(const SequenceSampler& _sampler, const uint32_t seed, const uint32_t _dimension) : sampler(_sampler), dimension(_dimension),
					mersenneTwister(std::hash<uint64_t>()((uint64_t(_dimension)<<32)|seed))
				{
					cachedFlip.resize(MAX_SAMPLES-1u);
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
				}
				inline uint32_t getTreeDepth(uint32_t sampleNum)
				{
					return hlsl::findMSB(sampleNum+1u);
				}

				const SequenceSampler& sampler;
				std::mt19937 mersenneTwister;
				core::vector<uint32_t> cachedFlip;
				const uint32_t dimension;
		};
		inline SDimensionSampler prepareDimension(const uint64_t dim) const
		{
			return SDimensionSampler(*this,seed,dim);
		}

	private:
		uint32_t seed;
};


}
#endif
