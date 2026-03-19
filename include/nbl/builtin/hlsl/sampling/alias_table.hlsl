// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_ALIAS_TABLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

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
// Template parameters are ReadOnly accessors, each with:
//   value_type get(uint32_t i) const;
//
// - ProbabilityAccessor: returns scalar_type threshold in [0, 1] for bin i
// - AliasIndexAccessor:  returns uint32_t redirect index for bin i
// - PdfAccessor:         returns scalar_type weight[i] / totalWeight
//
// Satisfies TractableSampler (not BackwardTractableSampler: the mapping is discrete).
// The cache stores the sampled index so forwardPdf can look up the PDF.
template<typename T, typename ProbabilityAccessor, typename AliasIndexAccessor, typename PdfAccessor>
struct AliasTable
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

	static AliasTable create(NBL_CONST_REF_ARG(ProbabilityAccessor) _probAccessor, NBL_CONST_REF_ARG(AliasIndexAccessor) _aliasAccessor, NBL_CONST_REF_ARG(PdfAccessor) _pdfAccessor, uint32_t _size)
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
		const uint32_t bin = uint32_t(scaled);
		const scalar_type remainder = scaled - scalar_type(bin);

		// Use if-statement to avoid select: aliasIndex is a dependent read
		codomain_type result;
		if (remainder < probAccessor.get(bin))
			result = bin;
		else
			result = aliasAccessor.get(bin);

		return result;
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

	ProbabilityAccessor probAccessor;
	AliasIndexAccessor aliasAccessor;
	PdfAccessor pdfAccessor;
	scalar_type tableSizeMinusUlp;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
