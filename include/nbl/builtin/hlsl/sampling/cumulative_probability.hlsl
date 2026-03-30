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

// Discrete sampler using cumulative probability lookup via upper_bound.
//
// Samples a discrete index in [0, N) with probability proportional to
// precomputed weights in O(log N) time per sample.
//
// The cumulative probability array stores N-1 entries (the last bucket
// is always 1.0 and need not be stored). Entry i holds the sum of
// probabilities for indices [0, i].
//
// Satisfies TractableSampler and ResamplableSampler (not BackwardTractableSampler:
// the mapping is discrete).
template<typename T, typename Domain, typename Codomain, typename CumProbAccessor
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

	static CumulativeProbabilitySampler create(NBL_CONST_REF_ARG(CumProbAccessor) _cumProbAccessor, uint32_t _size)
	{
		CumulativeProbabilitySampler retval;
		retval.cumProbAccessor = _cumProbAccessor;
		retval.storedCount = _size - 1u;
		return retval;
	}

	// BasicSampler interface
	codomain_type generate(const domain_type u) NBL_CONST_MEMBER_FUNC
	{
		// upper_bound returns first index where cumProb > u
		return hlsl::upper_bound(cumProbAccessor, 0u, storedCount, u);
	}

	// TractableSampler interface
	codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		// Stateful comparator that tracks the CDF values seen during binary search.
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
		const codomain_type result = hlsl::upper_bound(cumProbAccessor, 0u, storedCount, u, comp);
		cache.oneBefore = comp.oneBefore;
		cache.upperBound = comp.upperBound;
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

	weight_type backwardWeight(const codomain_type v) NBL_CONST_MEMBER_FUNC
	{
		return backwardPdf(v);
	}

	CumProbAccessor cumProbAccessor;
	uint32_t storedCount;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
