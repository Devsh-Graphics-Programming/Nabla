// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CUMULATIVE_PROBABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CUMULATIVE_PROBABILITY_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>

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
// Template parameters are ReadOnly accessors:
// - CumulativeProbabilityAccessor: returns scalar_type cumProb for index i,
//     must have `value_type` typedef and `operator[](uint32_t)` for upper_bound
// - PdfAccessor: returns scalar_type weight[i] / totalWeight
//
// Satisfies TractableSampler and ResamplableSampler (not BackwardTractableSampler:
// the mapping is discrete).
template<typename T, typename CumulativeProbabilityAccessor, typename PdfAccessor>
struct CumulativeProbabilitySampler
{
	using scalar_type = T;

	using domain_type = scalar_type;
	using codomain_type = uint32_t;
	using density_type = scalar_type;
	using weight_type = density_type;

	struct cache_type
	{
		codomain_type sampledIndex;
	};

	static CumulativeProbabilitySampler create(NBL_CONST_REF_ARG(CumulativeProbabilityAccessor) _cumProbAccessor, NBL_CONST_REF_ARG(PdfAccessor) _pdfAccessor, uint32_t _size)
	{
		CumulativeProbabilitySampler retval;
		retval.cumProbAccessor = _cumProbAccessor;
		retval.pdfAccessor = _pdfAccessor;
		retval.size = _size;
		return retval;
	}

	// BasicSampler interface
	codomain_type generate(const domain_type u)
	{
		// upper_bound on N-1 stored entries; if u >= all stored values, returns N-1 (the last bucket)
		const uint32_t storedCount = size - 1u;
		// upper_bound returns first index where cumProb > u
		return hlsl::upper_bound(cumProbAccessor, 0u, storedCount, u);
	}

	// TractableSampler interface
	codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
	{
		const codomain_type result = generate(u);
		cache.sampledIndex = result;
		return result;
	}

	density_type forwardPdf(NBL_CONST_REF_ARG(cache_type) cache)
	{
		return pdfAccessor.get(cache.sampledIndex);
	}

	weight_type forwardWeight(NBL_CONST_REF_ARG(cache_type) cache)
	{
		return forwardPdf(cache);
	}

	density_type backwardPdf(const codomain_type v)
	{
		return pdfAccessor.get(v);
	}

	weight_type backwardWeight(const codomain_type v)
	{
		return backwardPdf(v);
	}

	CumulativeProbabilityAccessor cumProbAccessor;
	PdfAccessor pdfAccessor;
	uint32_t size;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
