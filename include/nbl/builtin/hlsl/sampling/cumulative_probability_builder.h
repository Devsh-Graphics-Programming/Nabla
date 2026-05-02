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

// Returns the next power of two >= x (and 1 for x <= 1). Matches the leaf-count
// the Eytzinger builder pads to.
inline uint32_t eytzingerLeafCount(uint32_t N)
{
	uint32_t P = 1u;
	while (P < N) P <<= 1u;
	return P;
}

// Builds an Eytzinger-layout CDF for cache-friendly binary search on the GPU.
//
// Layout (1-indexed, size 2*P where P = eytzingerLeafCount(N)):
//   tree[0]           unused (keeps parent/child arithmetic branch-free)
//   tree[1 .. P-1]    interior split keys; tree[v] == rightmost leaf of v's left subtree
//   tree[P .. P+N-1]  leaves, tree[P + i] = normalized inclusive scan of weights up to i
//   tree[P+N .. 2P-1] padded leaves, all 1.0 (any u < 1.0 routes away from these)
//
// The sampler walks the tree as index = (index << 1) | goRight for ceil(log2(N))
// iterations. Successive taps within one search land on adjacent memory, so every
// cache line pulled is fully used and the first log2(subgroupSize) iterations are
// served by a single memory transaction per subgroup.
template<typename T>
void buildEytzinger(std::span<const T> weights, T* outTree)
{
	const uint32_t N = static_cast<uint32_t>(weights.size());
	if (N == 0)
		return;

	const uint32_t P = eytzingerLeafCount(N);

	T total = T(0);
	for (uint32_t i = 0; i < N; ++i)
		total += weights[i];
	const T rcpTotal = T(1) / total;

	T acc = T(0);
	for (uint32_t i = 0; i < N; ++i)
	{
		acc += weights[i];
		outTree[P + i] = acc * rcpTotal;
	}
	for (uint32_t i = N; i < P; ++i)
		outTree[P + i] = T(1);

	// Bottom-up: each interior node copies the rightmost leaf of its left subtree,
	// found by descending left-then-always-right from v.
	for (uint32_t v = P; v-- > 1u;)
	{
		uint32_t r = v << 1u;
		while (r < P)
			r = (r << 1u) | 1u;
		outTree[v] = outTree[r];
	}
}

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
