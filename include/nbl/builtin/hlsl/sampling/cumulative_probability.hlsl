// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CUMULATIVE_PROBABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CUMULATIVE_PROBABILITY_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

namespace concepts
{

// clang-format off
#define NBL_CONCEPT_NAME CumulativeProbabilityAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)(typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)(Scalar)
#define NBL_CONCEPT_PARAM_0 (accessor, T)
#define NBL_CONCEPT_PARAM_1 (index, uint32_t)
NBL_CONCEPT_BEGIN(2)
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define index NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
NBL_CONCEPT_END(
	((NBL_CONCEPT_REQ_TYPE)(T::value_type))
	((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor[index]), ::nbl::hlsl::is_same_v, Scalar)));
#undef index
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
// clang-format on

} // namespace concepts

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
template<typename T, typename CumProbAccessor NBL_FUNC_REQUIRES(concepts::CumulativeProbabilityAccessor<CumProbAccessor, T>)
struct CumulativeProbabilitySampler
{
	using scalar_type = T;

	using domain_type = scalar_type;
	using codomain_type = uint32_t;
	using density_type = scalar_type;
	using weight_type = density_type;

	struct cache_type
	{
		density_type oneBefore;
		density_type upperBound;
	};

	// Stateful comparator that tracks the CDF values seen during binary search.
	// upper_bound uses lower_to_upper_comparator_transform_t which calls !comp(rhs, lhs),
	// so our operator() receives (value=u, rhs=cumProb[testPoint]).
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
	};

	static CumulativeProbabilitySampler create(NBL_CONST_REF_ARG(CumProbAccessor) _cumProbAccessor, uint32_t _size)
	{
		CumulativeProbabilitySampler retval;
		retval.cumProbAccessor = _cumProbAccessor;
		retval.storedCount = _size - 1u;
		return retval;
	}

	// BasicSampler interface
	codomain_type generate(const domain_type u)
	{
		// upper_bound returns first index where cumProb > u
		return hlsl::upper_bound(cumProbAccessor, 0u, storedCount, u);
	}

	// TractableSampler interface
	codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
	{
		CdfComparator comp;
		comp.oneBefore = density_type(0.0);
		comp.upperBound = density_type(1.0);
		const codomain_type result = hlsl::upper_bound(cumProbAccessor, 0u, storedCount, u, comp);
		cache.oneBefore = comp.oneBefore;
		cache.upperBound = comp.upperBound;
		return result;
	}

	density_type forwardPdf(NBL_CONST_REF_ARG(cache_type) cache)
	{
		return cache.upperBound - cache.oneBefore;
	}

	weight_type forwardWeight(NBL_CONST_REF_ARG(cache_type) cache)
	{
		return forwardPdf(cache);
	}

	density_type backwardPdf(const codomain_type v)
	{
		const density_type cur = (v < storedCount) ? cumProbAccessor[v] : density_type(1.0);
		const density_type prev = (v > 0u) ? cumProbAccessor[v - 1u] : density_type(0.0);
		return cur - prev;
	}

	weight_type backwardWeight(const codomain_type v)
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
