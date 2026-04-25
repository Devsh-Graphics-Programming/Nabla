// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CUMULATIVE_PROBABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CUMULATIVE_PROBABILITY_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// Discrete sampler using cumulative probability lookup.
//
// Samples a discrete index in [0, N) with probability proportional to
// precomputed weights in O(log N) time per sample.
//
// Three layouts / cache-population strategies, selected by the Mode parameter:
//
//   TRACKING (default):  N-1 CDF entries, last bucket implicit at 1.0.
//                        A stateful comparator records the straddling CDF
//                        values during upper_bound itself.
//   YOLO:                Same storage. Plain upper_bound followed by two
//                        re-reads of the adjacent CDF entries (warm cache).
//                        Lower register footprint, two extra array reads.
//   EYTZINGER:           Level-order implicit binary tree in 2*P entries
//                        where P = roundUpPot(N). Leaves at [P, P+N) hold
//                        the CDF; interior nodes at [1, P) hold split keys.
//                        Descent reads adjacent memory at each step, so
//                        every cache line pulled is fully utilised and the
//                        first log2(subgroupSize) iterations are served by a
//                        single transaction per subgroup. Build with
//                        sampling::buildEytzinger<T>().
//
// Satisfies TractableSampler and ResamplableSampler (not BackwardTractableSampler:
// the mapping is discrete).
enum CumulativeProbabilityMode : uint32_t
{
	TRACKING  = 0u,
	YOLO      = 1u,
	EYTZINGER = 2u
};

template<typename T, typename Domain, typename Codomain, typename CumProbAccessor, CumulativeProbabilityMode Mode = CumulativeProbabilityMode::TRACKING
	NBL_PRIMARY_REQUIRES(concepts::accessors::GenericReadAccessor<CumProbAccessor, T, Codomain>)
struct CumulativeProbabilitySampler
{
	using scalar_type = T;

	using domain_type = Domain;
	using codomain_type = Codomain;
	using density_type = scalar_type;
	using weight_type = density_type;

	struct cache_type
	{
		density_type oneBefore;
		density_type upperBound;
	};

	// `_size` is the user-facing bucket count N for every mode. TRACKING / YOLO
	// expect the accessor to hold N-1 CDF entries; EYTZINGER expects 2*P entries
	// in the level-order layout produced by buildEytzinger.
	static CumulativeProbabilitySampler create(NBL_CONST_REF_ARG(CumProbAccessor) _cumProbAccessor, uint32_t _size)
	{
		CumulativeProbabilitySampler retval;
		retval.cumProbAccessor = _cumProbAccessor;
		retval.storedCount = _size - 1u;
		retval.depth = 0u;
		NBL_IF_CONSTEXPR(Mode == CumulativeProbabilityMode::EYTZINGER)
		{
			uint32_t P = 1u;
			uint32_t d = 0u;
			while (P < _size) { P <<= 1u; ++d; }
			retval.depth = d;
		}
		return retval;
	}

	// BasicSampler interface
	codomain_type generate(const domain_type u) NBL_CONST_MEMBER_FUNC
	{
		NBL_IF_CONSTEXPR(Mode == CumulativeProbabilityMode::EYTZINGER)
		{
			const uint32_t leafBase = 1u << depth;
			uint32_t index = 1u;
			for (uint32_t iter = 0u; iter < depth; ++iter)
			{
				density_type key;
				cumProbAccessor.template get<density_type, uint32_t>(index, key);
				index = (index << 1u) | uint32_t(!(u < key));
			}
			const codomain_type result = codomain_type(index - leafBase);
			return result < codomain_type(storedCount) ? result : codomain_type(storedCount);
		}
		else
		{
			return hlsl::upper_bound(cumProbAccessor, 0u, storedCount, u);
		}
	}

	// TractableSampler interface
	codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		codomain_type result;
		NBL_IF_CONSTEXPR(Mode == CumulativeProbabilityMode::EYTZINGER)
		{
			// Descent visits one interior node per level. Going left tightens
			// the upper bound to the current key; going right tightens the
			// lower bound. Final index, leafBase is the bucket.
			cache.oneBefore = density_type(0.0);
			cache.upperBound = density_type(1.0);
			const uint32_t leafBase = 1u << depth;
			uint32_t index = 1u;
			for (uint32_t iter = 0u; iter < depth; ++iter)
			{
				density_type key;
				cumProbAccessor.template get<density_type, uint32_t>(index, key);
				const bool goRight = !(u < key);
				if (goRight)
				{
					cache.oneBefore = key;
					index = (index << 1u) | 1u;
				}
				else
				{
					cache.upperBound = key;
					index = (index << 1u);
				}
			}
			const codomain_type raw = codomain_type(index - leafBase);
			result = raw < codomain_type(storedCount) ? raw : codomain_type(storedCount);
		}
		else NBL_IF_CONSTEXPR(Mode == CumulativeProbabilityMode::YOLO)
		{
			// Re-read the two adjacent CDF entries after the binary search.
			// Both sit on the cache lines the search just touched, so they are warm.
			result = hlsl::upper_bound(cumProbAccessor, 0u, storedCount, u);
			cache.oneBefore = density_type(0.0);
			if (result)
				cumProbAccessor.template get<density_type, codomain_type>(result - 1u, cache.oneBefore);
			cache.upperBound = density_type(1.0);
			if (result < storedCount)
				cumProbAccessor.template get<density_type, codomain_type>(result, cache.upperBound);
		}
		else
		{
			// TRACKING: stateful comparator captures the CDF values straddling the
			// found index during the binary search itself, avoiding the two extra reads.
			struct CdfComparator
			{
				bool operator()(const density_type value, const density_type rhs)
				{
					const bool retval = value < rhs;
					if (retval)
						upperBound = rhs;
					else
						oneBefore = rhs;
					return retval;
				}

				density_type oneBefore;
				density_type upperBound;
			} comp;
			comp.oneBefore = density_type(0.0);
			comp.upperBound = density_type(1.0);
			result = hlsl::upper_bound(cumProbAccessor, 0u, storedCount, u, comp);
			cache.oneBefore = comp.oneBefore;
			cache.upperBound = comp.upperBound;
		}
		return result;
	}

	density_type forwardPdf(const domain_type u, NBL_CONST_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		return cache.upperBound - cache.oneBefore;
	}

	weight_type forwardWeight(const domain_type u, NBL_CONST_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		return forwardPdf(u, cache);
	}

	density_type backwardPdf(const codomain_type v) NBL_CONST_MEMBER_FUNC
	{
		NBL_IF_CONSTEXPR(Mode == CumulativeProbabilityMode::EYTZINGER)
		{
			// Leaves store the CDF directly; the last real leaf is normalized
			// to 1.0 and padded leaves (if any) also hold 1.0.
			const uint32_t leafBase = 1u << depth;
			density_type retval;
			cumProbAccessor.template get<density_type, uint32_t>(leafBase + uint32_t(v), retval);
			if (v)
			{
				density_type prev;
				cumProbAccessor.template get<density_type, uint32_t>(leafBase + uint32_t(v) - 1u, prev);
				retval -= prev;
			}
			return retval;
		}
		else
		{
			density_type retval = density_type(1.0);
			if (v < storedCount)
				cumProbAccessor.template get<density_type, codomain_type>(v, retval);
			if (v)
			{
				density_type prev;
				cumProbAccessor.template get<density_type, codomain_type>(v - 1u, prev);
				retval -= prev;
			}
			return retval;
		}
	}

	weight_type backwardWeight(const codomain_type v) NBL_CONST_MEMBER_FUNC
	{
		return backwardPdf(v);
	}

	CumProbAccessor cumProbAccessor;
	uint32_t storedCount;    // N - 1 (last real bucket index)
	uint32_t depth;          // EYTZINGER only: ceil(log2(N)), iteration count; leafBase = 1 << depth
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
