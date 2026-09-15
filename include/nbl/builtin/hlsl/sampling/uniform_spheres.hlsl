// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_SPHERES_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_SPHERES_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/sampling/quotient_and_pdf.hlsl"
#include "nbl/builtin/hlsl/sampling/concentric_mapping.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct UniformHemisphere
{
	using vector_t2 = vector<T, 2>;
	using vector_t3 = vector<T, 3>;

	// BijectiveSampler concept types
	using scalar_type = T;
	using domain_type = vector_t2;
	using codomain_type = vector_t3;
	using density_type = T;
	using weight_type = density_type;

	struct cache_type {};

	static codomain_type generate(const domain_type u)
	{
		typename ConcentricMapping<T>::cache_type cmCache;
		const vector_t2 p = ConcentricMapping<T>::generate(u, cmCache);
		const T z = T(1.0) - cmCache.r2;
		const T xyScale = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(2.0) - cmCache.r2));
		return vector_t3(p.x * xyScale, p.y * xyScale, z);
	}

	static codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
	{
		return generate(u);
	}

	static domain_type generateInverse(const codomain_type v)
	{
		// r_disk / r_xy = sqrt(1-z) / sqrt(1-z^2) = 1/sqrt(1+z)
		const T scale = T(1.0) / hlsl::sqrt<T>(T(1.0) + v.z);
		return ConcentricMapping<T>::generateInverse(vector_t2(v.x * scale, v.y * scale));
	}

	static density_type forwardPdf(const domain_type u, const cache_type cache)
	{
		return T(0.5) * numbers::inv_pi<T>;
	}

	static weight_type forwardWeight(const domain_type u, const cache_type cache)
	{
		return T(0.5) * numbers::inv_pi<T>;
	}

	static density_type backwardPdf(const codomain_type v)
	{
		assert(v.z > 0);
		return T(0.5) * numbers::inv_pi<T>;
	}

	static weight_type backwardWeight(const codomain_type v)
	{
		assert(v.z > 0);
		return T(0.5) * numbers::inv_pi<T>;
	}

};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct UniformSphere
{
	using vector_t2 = vector<T, 2>;
	using vector_t3 = vector<T, 3>;
	using hemisphere_t = UniformHemisphere<T>;

	// BijectiveSampler concept types
	using scalar_type = T;
	using domain_type = vector_t2;
	using codomain_type = vector_t3;
	using density_type = T;
	using weight_type = density_type;

	using cache_type = typename hemisphere_t::cache_type;

	static codomain_type generate(const domain_type u)
	{
		const T tmp = u.x * T(2.0) - T(1.0);
		const codomain_type L = hemisphere_t::generate(vector_t2(hlsl::abs<T>(tmp), u.y));
		return vector_t3(L.x, L.y, L.z * hlsl::sign(tmp));
	}

	static codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
	{
		return generate(u);
	}

	static domain_type generateInverse(const codomain_type v)
	{
		const T dir = hlsl::sign(v.z) * T(0.5);
		const domain_type hemiU = hemisphere_t::generateInverse(vector_t3(v.x, v.y, hlsl::abs<T>(v.z)));
		return vector_t2(hemiU.x * dir + T(0.5), hemiU.y);
	}

	static density_type forwardPdf(const domain_type u, const cache_type cache)
	{
		return T(0.5) * hemisphere_t::forwardPdf(u, cache);
	}

	static weight_type forwardWeight(const domain_type u, const cache_type cache)
	{
		return T(0.5) * hemisphere_t::forwardWeight(u, cache);
	}

	static density_type backwardPdf(const codomain_type v)
	{
		return T(0.25) * numbers::inv_pi<T>;
	}

	static weight_type backwardWeight(const codomain_type v)
	{
		return T(0.25) * numbers::inv_pi<T>;
	}

};
} // namespace sampling

} // namespace hlsl
} // namespace nbl

#endif
