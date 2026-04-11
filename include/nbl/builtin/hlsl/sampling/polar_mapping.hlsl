// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_POLAR_MAPPING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_POLAR_MAPPING_INCLUDED_

#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct PolarMapping
{
	using scalar_type = T;
	using vector2_type = vector<T, 2>;

	// BijectiveSampler concept types
	using domain_type = vector2_type;
	using codomain_type = vector2_type;
	using density_type = scalar_type;
	using weight_type = density_type;

	struct cache_type {};

	static codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
	{
		const scalar_type r = hlsl::sqrt<scalar_type>(u.x);
		const scalar_type phi = scalar_type(2) * numbers::pi<scalar_type> * u.y;
		return vector2_type(r * hlsl::cos<scalar_type>(phi), r * hlsl::sin<scalar_type>(phi));
	}

	static codomain_type generate(const domain_type u)
	{
		cache_type dummy;
		return generate(u, dummy);
	}

	static domain_type generateInverse(const codomain_type p)
	{
		const scalar_type r2 = p.x * p.x + p.y * p.y;
		scalar_type phi = hlsl::atan2(p.y, p.x);
		phi += hlsl::mix(scalar_type(0), scalar_type(2) * numbers::pi<scalar_type>, phi < scalar_type(0));
		return vector2_type(r2, phi * (scalar_type(0.5) * numbers::inv_pi<scalar_type>));
	}

	static density_type forwardPdf(const domain_type u, cache_type cache) { return numbers::inv_pi<scalar_type>; }
	static density_type backwardPdf(codomain_type v) { return numbers::inv_pi<scalar_type>; }

	static weight_type forwardWeight(const domain_type u, cache_type cache) { return forwardPdf(u, cache); }
	static weight_type backwardWeight(codomain_type v) { return backwardPdf(v); }
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
