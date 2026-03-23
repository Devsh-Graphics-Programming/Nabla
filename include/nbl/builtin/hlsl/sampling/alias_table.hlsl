// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>
#include <nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// Alias Method (Vose/Walker) discrete sampler.
//
// Samples a discrete index in [0, N) with probability proportional to
// precomputed weights in O(1) time per sample, using a prebuilt alias table.
//
// Accessor template parameters must satisfy GenericReadAccessor:
//   accessor.template get<V, I>(index, outVal)  // void, writes to outVal
//
// - ProbabilityAccessor: reads scalar_type threshold in [0, 1] for bin i
// - AliasIndexAccessor:  reads uint32_t redirect index for bin i
// - PdfAccessor:         reads scalar_type weight[i] / totalWeight
//
// Satisfies TractableSampler (not BackwardTractableSampler: the mapping is discrete).
// The cache stores the PDF value looked up during generate, avoiding redundant
// storage of the codomain (sampled index) which is already the return value.
template<typename T, typename Domain, typename Codomain, typename ProbabilityAccessor, typename AliasIndexAccessor, typename PdfAccessor
	NBL_PRIMARY_REQUIRES(
		concepts::accessors::GenericReadAccessor<ProbabilityAccessor, T, Codomain> &&
		concepts::accessors::GenericReadAccessor<AliasIndexAccessor, Codomain, Codomain> &&
		concepts::accessors::GenericReadAccessor<PdfAccessor, T, Codomain>)
struct AliasTable
{
	using scalar_type = T;

	using domain_type = Domain;
	using codomain_type = Codomain;
	using density_type = scalar_type;
	using weight_type = density_type;

	struct cache_type
	{
		density_type pdf;
	};

	static AliasTable create(NBL_CONST_REF_ARG(ProbabilityAccessor) _probAccessor, NBL_CONST_REF_ARG(AliasIndexAccessor) _aliasAccessor, NBL_CONST_REF_ARG(PdfAccessor) _pdfAccessor, codomain_type _size)
	{
		AliasTable retval;
		retval.probAccessor = _probAccessor;
		retval.aliasAccessor = _aliasAccessor;
		retval.pdfAccessor = _pdfAccessor;
		// Precompute tableSize as float minus 1 ULP so that u=1.0 maps to bin N-1
		const scalar_type exact = scalar_type(_size);
		retval.tableSizeMinusUlp = nbl::hlsl::bit_cast<scalar_type>(nbl::hlsl::bit_cast<uint32_t>(exact) - 1u);
		return retval;
	}

	// BasicSampler interface
	codomain_type generate(const domain_type u)
	{
		const scalar_type scaled = u * tableSizeMinusUlp;
		const codomain_type bin = codomain_type(scaled);
		const scalar_type remainder = scaled - scalar_type(bin);

		scalar_type prob;
		probAccessor.template get<scalar_type, codomain_type>(bin, prob);

		// Use if-statement to avoid select: aliasIndex is a dependent read
		codomain_type result;
		if (remainder < prob)
		{
			result = bin;
		}
		else
		{
			codomain_type alias;
			aliasAccessor.template get<codomain_type, codomain_type>(bin, alias);
			result = alias;
		}

		return result;
	}

	// TractableSampler interface
	codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
	{
		const codomain_type result = generate(u);
		pdfAccessor.template get<scalar_type, codomain_type>(result, cache.pdf);
		return result;
	}

	density_type forwardPdf(NBL_CONST_REF_ARG(cache_type) cache)
	{
		return cache.pdf;
	}

	weight_type forwardWeight(NBL_CONST_REF_ARG(cache_type) cache)
	{
		return cache.pdf;
	}

	density_type backwardPdf(const codomain_type v)
	{
		scalar_type pdf;
		pdfAccessor.template get<scalar_type, codomain_type>(v, pdf);
		return pdf;
	}

	weight_type backwardWeight(const codomain_type v)
	{
		return backwardPdf(v);
	}

	ProbabilityAccessor probAccessor;
	AliasIndexAccessor aliasAccessor;
	PdfAccessor pdfAccessor;
	scalar_type tableSizeMinusUlp;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
