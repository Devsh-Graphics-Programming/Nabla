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

	static codomain_type __generate(const domain_type _sample)
	{
		T z = _sample.x;
		T r = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(1.0) - z * z));
		T phi = T(2.0) * numbers::pi<T> * _sample.y;
		return vector_t3(r * hlsl::cos<T>(phi), r * hlsl::sin<T>(phi), z);
	}

	static codomain_type generate(const domain_type _sample)
	{
		return __generate(_sample);
	}

	static codomain_type generate(const domain_type _sample, NBL_REF_ARG(cache_type) cache)
	{
		return __generate(_sample);
	}

	static domain_type __generateInverse(const codomain_type _sample)
	{
		T phi = hlsl::atan2(_sample.y, _sample.x);
		const T twopi = T(2.0) * numbers::pi<T>;
		phi += hlsl::mix(T(0.0), twopi, phi < T(0.0));
		return vector_t2(_sample.z, phi / twopi);
	}

	static domain_type generateInverse(const codomain_type _sample)
	{
		return __generateInverse(_sample);
	}

	static density_type forwardPdf(const cache_type cache)
	{
		return T(0.5) * numbers::inv_pi<T>;
	}

	static weight_type forwardWeight(const cache_type cache)
	{
		return T(0.5) * numbers::inv_pi<T>;
	}

	static density_type backwardPdf(const vector_t3 _sample)
	{
		return T(0.5) * numbers::inv_pi<T>;
	}

	static weight_type backwardWeight(const codomain_type sample)
	{
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

	static codomain_type __generate(const domain_type _sample)
	{
		// Map _sample.x from [0,1] into hemisphere sample + sign flip:
		// upper hemisphere when _sample.x < 0.5, lower when >= 0.5
		const bool chooseLower = _sample.x >= T(0.5);
		const T hemiX = chooseLower ? (T(2.0) * _sample.x - T(1.0)) : (T(2.0) * _sample.x);
		vector_t3 retval = hemisphere_t::__generate(vector_t2(hemiX, _sample.y));
		retval.z = chooseLower ? (-retval.z) : retval.z;
		return retval;
	}

	static codomain_type generate(const domain_type _sample)
	{
		return __generate(_sample);
	}

	static codomain_type generate(const domain_type _sample, NBL_REF_ARG(cache_type) cache)
	{
		return __generate(_sample);
	}

	static domain_type __generateInverse(const codomain_type _sample)
	{
		const bool isLower = _sample.z < T(0.0);
		const vector_t3 hemiSample = vector_t3(_sample.x, _sample.y, hlsl::abs(_sample.z));
		vector_t2 hemiUV = hemisphere_t::__generateInverse(hemiSample);
		// Recover _sample.x: upper hemisphere maps [0,0.5], lower maps [0.5,1]
		hemiUV.x = isLower ? (hemiUV.x * T(0.5) + T(0.5)) : (hemiUV.x * T(0.5));
		return hemiUV;
	}

	static domain_type generateInverse(const codomain_type _sample)
	{
		return __generateInverse(_sample);
	}

	static density_type forwardPdf(const cache_type cache)
	{
		return T(0.5) * hemisphere_t::forwardPdf(cache);
	}

	static weight_type forwardWeight(const cache_type cache)
	{
		return T(0.5) * hemisphere_t::forwardWeight(cache);
	}

	static density_type backwardPdf(const vector_t3 _sample)
	{
		return T(0.5) * hemisphere_t::backwardPdf(_sample);
	}

	static weight_type backwardWeight(const codomain_type sample)
	{
		return T(0.5) * hemisphere_t::backwardWeight(sample);
	}

};
} // namespace sampling

} // namespace hlsl
} // namespace nbl

#endif
