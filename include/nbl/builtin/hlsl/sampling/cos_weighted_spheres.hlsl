// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_SPHERES_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_COS_WEIGHTED_SPHERES_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/sampling/concentric_mapping.hlsl"
#include "nbl/builtin/hlsl/sampling/quotient_and_pdf.hlsl"

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
		density_type pdf;
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
		cache.pdf = __pdf(L.z);
		return L;
	}

	static domain_type __generateInverse(const codomain_type L)
	{
		return ConcentricMapping<T>::generateInverse(L.xy);
	}

	static domain_type generateInverse(const codomain_type L)
	{
		return __generateInverse(L);
	}

	static T __pdf(const T L_z)
	{
		return L_z * numbers::inv_pi<T>;
	}

	static scalar_type pdf(const T L_z)
	{
		return __pdf(L_z);
	}

	static density_type forwardPdf(const cache_type cache)
	{
		return cache.pdf;
	}

	static weight_type forwardWeight(const cache_type cache)
	{
		return forwardPdf(cache);
	}

	static density_type backwardPdf(const codomain_type L)
	{
		return __pdf(L.z);
	}

	static weight_type backwardWeight(const codomain_type L)
	{
		return backwardPdf(L);
	}

	template<typename U = vector<T, 1> >
	static quotient_and_pdf<U, T> quotientAndPdf(const T L)
	{
		return quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), __pdf(L));
	}

	template<typename U = vector<T, 1> >
	static quotient_and_pdf<U, T> quotientAndPdf(const vector_t3 L)
	{
		return quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), __pdf(L.z));
	}
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct ProjectedSphere
{
	using vector_t2 = vector<T, 2>;
	using vector_t3 = vector<T, 3>;
	using hemisphere_t = ProjectedHemisphere<T>;

	// BijectiveSampler concept types
	using scalar_type = T;
	using domain_type = vector_t3;
	using codomain_type = vector_t3;
	using density_type = T;
	using weight_type = density_type;

	struct cache_type
	{
		density_type pdf;
	};

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
		cache.pdf = __pdf(L.z);
		return L;
	}

	static domain_type __generateInverse(const codomain_type L)
	{
		// NOTE: incomplete information to recover exact z component; we only know which hemisphere L came from,
		// so we return a canonical value (0.0 for upper, 1.0 for lower) that round-trips correctly through __generate
		return vector_t3(hemisphere_t::__generateInverse(L), hlsl::mix(T(1.0), T(0.0), L.z > T(0.0)));
	}

	static domain_type generateInverse(const codomain_type L)
	{
		return __generateInverse(L);
	}

	static T __pdf(T L_z)
	{
		return T(0.5) * hemisphere_t::__pdf(hlsl::abs(L_z));
	}

	static scalar_type pdf(T L_z)
	{
		return __pdf(L_z);
	}

	static density_type forwardPdf(const cache_type cache)
	{
		return cache.pdf;
	}

	static weight_type forwardWeight(const cache_type cache)
	{
		return forwardPdf(cache);
	}

	static density_type backwardPdf(const codomain_type L)
	{
		return __pdf(L.z);
	}

	static weight_type backwardWeight(const codomain_type L)
	{
		return backwardPdf(L);
	}

	template<typename U = vector<T, 1> >
	static quotient_and_pdf<U, T> quotientAndPdf(T L)
	{
		return quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), __pdf(L));
	}

	template<typename U = vector<T, 1> >
	static quotient_and_pdf<U, T> quotientAndPdf(const vector_t3 L)
	{
		return quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), __pdf(L.z));
	}
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
