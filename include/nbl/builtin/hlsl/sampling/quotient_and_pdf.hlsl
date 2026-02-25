// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_PDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_QUOTIENT_AND_PDF_INCLUDED_

#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{
// Returned by TractableSampler::generate — codomain sample bundled with its rcpPdf
template<typename V, typename P>
struct codomain_and_rcpPdf
{
	using this_t = codomain_and_rcpPdf<V, P>;

	static this_t create(const V _value, const P _rcpPdf)
	{
		this_t retval;
		retval.value = _value;
		retval.rcpPdf = _rcpPdf;
		return retval;
	}

	V value;
	P rcpPdf;
};

// Returned by TractableSampler::generate — codomain sample bundled with its pdf
template<typename V, typename P>
struct codomain_and_pdf
{
	using this_t = codomain_and_pdf<V, P>;

	static this_t create(const V _value, const P _pdf)
	{
		this_t retval;
		retval.value = _value;
		retval.pdf = _pdf;
		return retval;
	}

	V value;
	P pdf;
};

// Returned by BijectiveSampler::invertGenerate — domain value bundled with its rcpPdf
template<typename V, typename P>
struct domain_and_rcpPdf
{
	using this_t = domain_and_rcpPdf<V, P>;

	static this_t create(const V _value, const P _rcpPdf)
	{
		this_t retval;
		retval.value = _value;
		retval.rcpPdf = _rcpPdf;
		return retval;
	}

	V value;
	P rcpPdf;
};

// Returned by BijectiveSampler::invertGenerate — domain value bundled with its pdf
template<typename V, typename P>
struct domain_and_pdf
{
	using this_t = domain_and_pdf<V, P>;

	static this_t create(const V _value, const P _pdf)
	{
		this_t retval;
		retval.value = _value;
		retval.pdf = _pdf;
		return retval;
	}

	V value;
	P pdf;
};

// finally fixed the semantic F-up, value/pdf = quotient not remainder
template<typename Q, typename P NBL_PRIMARY_REQUIRES(concepts::FloatingPointLikeVectorial<Q>&& concepts::FloatingPointLikeScalar<P>)
struct quotient_and_pdf
{
	using this_t = quotient_and_pdf<Q, P>;
	using scalar_q = typename vector_traits<Q>::scalar_type;

	static this_t create(const Q _quotient, const P _pdf)
	{
		this_t retval;
		retval.quotient = _quotient;
		retval.pdf = _pdf;
		return retval;
	}

	static this_t create(const scalar_q _quotient, const P _pdf)
	{
		this_t retval;
		retval.quotient = hlsl::promote<Q>(_quotient);
		retval.pdf = _pdf;
		return retval;
	}

	Q value()
	{
		return quotient * pdf;
	}

	Q quotient;
	P pdf;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
