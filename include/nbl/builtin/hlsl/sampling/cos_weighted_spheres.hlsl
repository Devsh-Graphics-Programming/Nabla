// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_SPHERES_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_SPHERES_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/sampling/concentric_mapping.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct ProjectedHemisphere
{
	using vector_t2 = vector<T, 2>;
	using vector_t3 = vector<T, 3>;

	// BijectiveSampler concept types
	using scalar_type = T;
	using domain_type = vector_t2;
	using codomain_type = vector_t3;
	using density_type = T;
	using weight_type = density_type;

	struct cache_type
	{
		scalar_type L_z;
	};

	static codomain_type __generate(const domain_type _sample)
	{
		vector_t2 p = ConcentricMapping<T>::generate(_sample * T(0.99999) + T(0.000005));
		T z = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(1.0) - p.x * p.x - p.y * p.y));
		return vector_t3(p.x, p.y, z);
	}

	static codomain_type generate(const domain_type _sample, NBL_REF_ARG(cache_type) cache)
	{
		const codomain_type L = __generate(_sample);
		cache.L_z = L.z;
		return L;
	}

	static domain_type generateInverse(const codomain_type L)
	{
		return ConcentricMapping<T>::generateInverse(L.xy);
	}

	static density_type forwardPdf(const cache_type cache)
	{
		return cache.L_z * numbers::inv_pi<T>;
	}

	static weight_type forwardWeight(const cache_type cache)
	{
		return forwardPdf(cache);
	}

	static density_type backwardPdf(const codomain_type L)
	{
		assert(L.z > 0);
		
		cache_type c;
		c.L_z = L.z;
		return forwardPdf(c);
	}

	static weight_type backwardWeight(const codomain_type L)
	{
		return backwardPdf(L);
	}
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct ProjectedSphere
{
	using vector_t2 = vector<T, 2>;
	using vector_t3 = vector<T, 3>;
	using hemisphere_t = ProjectedHemisphere<T>;

	// BackwardTractableSampler concept types (not bijective: z input is consumed for hemisphere selection)
	using scalar_type = T;
	using domain_type = vector_t3;
	using codomain_type = vector_t3;
	using density_type = T;
	using weight_type = density_type;

	using cache_type = typename hemisphere_t::cache_type;

	static codomain_type __generate(NBL_REF_ARG(domain_type) _sample)
	{
		vector_t3 retval = hemisphere_t::__generate(_sample.xy);
		const bool chooseLower = _sample.z > T(0.5);
		retval.z = chooseLower ? (-retval.z) : retval.z;
		if (chooseLower)
			_sample.z -= T(0.5);
		_sample.z *= T(2.0);
		return retval;
	}

	static codomain_type generate(NBL_REF_ARG(domain_type) _sample, NBL_REF_ARG(cache_type) cache)
	{
		const codomain_type L = __generate(_sample);
		cache.L_z = L.z;
		return L;
	}

	static density_type forwardPdf(const cache_type cache)
	{
		cache_type hc;
		hc.L_z = hlsl::abs(cache.L_z);
		return T(0.5) * hemisphere_t::forwardPdf(hc);
	}

	static weight_type forwardWeight(const cache_type cache)
	{
		return forwardPdf(cache);
	}

	static density_type backwardPdf(const codomain_type L)
	{
		cache_type c;
		c.L_z = L.z;
		return forwardPdf(c);
	}

	static weight_type backwardWeight(const codomain_type L)
	{
		return backwardPdf(L);
	}
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
