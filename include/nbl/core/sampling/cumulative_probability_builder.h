// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_CORE_SAMPLING_CUMULATIVE_PROBABILITY_BUILDER_H_INCLUDED_
#define _NBL_CORE_SAMPLING_CUMULATIVE_PROBABILITY_BUILDER_H_INCLUDED_

#include <cstdint>

namespace nbl
{
namespace core
{
namespace sampling
{

// Builds the CDF and PDF arrays from an array of non-negative weights.
//
// Parameters:
//   weights     - input weights (non-negative, at least one must be > 0)
//   N           - number of entries
//   outCumProb  - [out] cumulative probability array, N-1 entries
//                 (last bucket implicitly 1.0)
//   outPdf      - [out] normalized PDF per entry: weight[i] / sum(weights), N entries
template<typename T>
struct CumulativeProbabilityBuilder
{
	static void build(const T* weights, uint32_t N, T* outCumProb, T* outPdf)
	{
		T totalWeight = T(0);
		for (uint32_t i = 0; i < N; i++)
			totalWeight += weights[i];

		const T rcpTotalWeight = T(1) / totalWeight;

		for (uint32_t i = 0; i < N; i++)
			outPdf[i] = weights[i] * rcpTotalWeight;

		// N-1 stored entries (last bucket is implicitly 1.0)
		T cumulative = T(0);
		for (uint32_t i = 0; i < N - 1; i++)
		{
			cumulative += outPdf[i];
			outCumProb[i] = cumulative;
		}
	}
};

} // namespace sampling
} // namespace core
} // namespace nbl

#endif
