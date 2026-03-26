// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CUMULATIVE_PROBABILITY_BUILDER_H_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CUMULATIVE_PROBABILITY_BUILDER_H_INCLUDED_

#include <numeric>
#include <algorithm>
#include <span>
#include <cstdint>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// Builds a normalized cumulative histogram from an array of non-negative weights.
// Output has N-1 entries (last bucket implicitly 1.0).
template<typename T>
void computeNormalizedCumulativeHistogram(std::span<const T> weights, T* outCumProb)
{
	const auto N = weights.size();
	if (N < 2)
		return;
	std::inclusive_scan(weights.begin(), weights.end() - 1, outCumProb);
	const T normalizationFactor = T(1) / (outCumProb[N - 2] + weights[N - 1]);
	std::for_each(outCumProb, outCumProb + N - 1, [normalizationFactor](T& v) { v *= normalizationFactor; });
}

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
