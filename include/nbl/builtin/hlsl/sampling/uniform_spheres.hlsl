// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_SPHERES_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_SPHERES_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/sampling/quotient_and_pdf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

namespace impl
{
template<typename T>
vector<T, 3> directionFromZandPhi(const T z, const T phiSample)
{
	const T r = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(1.0) - z * z));
	const T phi = T(2.0) * numbers::pi<T> * phiSample;
	return vector<T, 3>(r * hlsl::cos<T>(phi), r * hlsl::sin<T>(phi), z);
}

template<typename T>
T phiSampleFromDirection(const T x, const T y)
{
	T phi = hlsl::atan2(y, x);
	const T twopi = T(2.0) * numbers::pi<T>;
	phi /= twopi;
	phi += hlsl::select(phi < T(0.0), T(1.0), T(0.0));
	return phi;
}
}

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

	static codomain_type __generate(const domain_type u)
	{
		return impl::directionFromZandPhi(u.x, u.y);
	}

	static codomain_type generate(const domain_type u)
	{
		return __generate(u);
	}

	static codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
	{
		return __generate(u);
	}

	static domain_type __generateInverse(const codomain_type v)
	{
		return vector_t2(v.z, impl::phiSampleFromDirection(v.x, v.y));
	}

	static domain_type generateInverse(const codomain_type v)
	{
		return __generateInverse(v);
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

	static codomain_type __generate(const domain_type u)
	{
		return impl::directionFromZandPhi(T(1.0) - T(2.0) * u.x, u.y);
	}

	static codomain_type generate(const domain_type u)
	{
		return __generate(u);
	}

	static codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
	{
		return __generate(u);
	}

	static domain_type __generateInverse(const codomain_type v)
	{
		// Inverse of z = 1 - 2*x => x = (1 - z) / 2
		return vector_t2((T(1.0) - v.z) * T(0.5), impl::phiSampleFromDirection(v.x, v.y));
	}

	static domain_type generateInverse(const codomain_type v)
	{
		return __generateInverse(v);
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
		return T(0.5) * hemisphere_t::backwardPdf(v);
	}

	static weight_type backwardWeight(const codomain_type v)
	{
		return T(0.5) * hemisphere_t::backwardWeight(v);
	}

};
} // namespace sampling

} // namespace hlsl
} // namespace nbl

#endif
