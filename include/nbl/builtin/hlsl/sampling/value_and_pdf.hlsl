// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_VALUE_AND_PDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_VALUE_AND_PDF_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename V, typename P>
struct value_and_rcpPdf
{
	using this_t = value_and_rcpPdf<V, P>;

	static this_t create(const V _value, const P _rcpPdf)
	{
		this_t retval;
		retval._value = _value;
		retval._rcpPdf = _rcpPdf;
		return retval;
	}

	V value() { return _value; }
	P rcpPdf() { return _rcpPdf; }

	V _value;
	P _rcpPdf;
};

template<typename V, typename P>
struct value_and_pdf
{
	using this_t = value_and_pdf<V, P>;

	static this_t create(const V _value, const P _pdf)
	{
		this_t retval;
		retval._value = _value;
		retval._pdf = _pdf;
		return retval;
	}

	V value() { return _value; }
	P pdf() { return _pdf; }

	V _value;
	P _pdf;
};

// Returned by TractableSampler::generate, codomain sample bundled with its rcpPdf
template<typename V, typename P>
using codomain_and_rcpPdf = value_and_rcpPdf<V, P>;

// Returned by TractableSampler::generate, codomain sample bundled with its pdf
template<typename V, typename P>
using codomain_and_pdf = value_and_pdf<V, P>;

// Returned by BijectiveSampler::invertGenerate, domain value bundled with its rcpPdf
template<typename V, typename P>
using domain_and_rcpPdf = value_and_rcpPdf<V, P>;

// Returned by BijectiveSampler::invertGenerate, domain value bundled with its pdf
template<typename V, typename P>
using domain_and_pdf = value_and_pdf<V, P>;

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
