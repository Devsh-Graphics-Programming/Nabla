// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_BUILDER_H_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_BUILDER_H_INCLUDED_

#include <cstdint>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// Builds the alias table from an array of non-negative weights.
// All output arrays must be pre-allocated to N entries.
//
// Parameters:
//   weights         - input weights (non-negative, at least one must be > 0)
//   N               - number of entries
//   outProbability  - [out] alias table probability threshold per bin, in [0, 1]
//   outAlias        - [out] alias redirect index per bin
//   outPdf          - [out] normalized PDF per entry: weight[i] / sum(weights)
//   workspace       - scratch buffer of N uint32_t entries
template<typename T>
struct AliasTableBuilder
{
	static void build(const T* weights, uint32_t N, T* outProbability, uint32_t* outAlias, T* outPdf, uint32_t* workspace)
	{
		T totalWeight = T(0);
		for (uint32_t i = 0; i < N; i++)
			totalWeight += weights[i];

		const T rcpTotalWeight = T(1) / totalWeight;

		// Compute PDFs, scaled probabilities, and partition into small/large in one pass
		uint32_t smallEnd = 0;
		uint32_t largeBegin = N;
		for (uint32_t i = 0; i < N; i++)
		{
			outPdf[i] = weights[i] * rcpTotalWeight;
			outProbability[i] = outPdf[i] * T(N);

			if (outProbability[i] < T(1))
				workspace[smallEnd++] = i;
			else
				workspace[--largeBegin] = i;
		}

		// Pair small and large entries
		while (smallEnd > 0 && largeBegin < N)
		{
			const uint32_t s = workspace[--smallEnd];
			const uint32_t l = workspace[largeBegin];

			outAlias[s] = l;
			// outProbability[s] already holds the correct probability for bin s

			outProbability[l] -= (T(1) - outProbability[s]);

			if (outProbability[l] < T(1))
			{
				// l became small: pop from large, push to small
				largeBegin++;
				workspace[smallEnd++] = l;
			}
			// else l stays in large (don't pop, reuse next iteration)
		}

		// Remaining entries (floating point rounding artifacts)
		while (smallEnd > 0)
		{
			const uint32_t s = workspace[--smallEnd];
			outProbability[s] = T(1);
			outAlias[s] = s;
		}
		while (largeBegin < N)
		{
			const uint32_t l = workspace[largeBegin++];
			outProbability[l] = T(1);
			outAlias[l] = l;
		}
	}
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
