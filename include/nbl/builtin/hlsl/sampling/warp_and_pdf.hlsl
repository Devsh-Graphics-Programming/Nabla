// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_WARP_AND_PDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_WARP_AND_PDF_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// Returned by TractableSampler::generate, codomain sample bundled with its rcpPdf
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

// Returned by TractableSampler::generate, codomain sample bundled with its pdf
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

// Returned by BijectiveSampler::invertGenerate, domain value bundled with its rcpPdf
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

// Returned by BijectiveSampler::invertGenerate, domain value bundled with its pdf
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

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
