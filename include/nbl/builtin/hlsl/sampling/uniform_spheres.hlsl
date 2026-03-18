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

	static scalar_type __pdf()
	{
		return T(1.0) / (T(2.0) * numbers::pi<T>);
	}

	static density_type forwardPdf(const cache_type cache)
	{
		return __pdf();
	}

	static weight_type forwardWeight(const cache_type cache)
	{
		return __pdf();
	}

	static density_type backwardPdf(const vector_t3 _sample)
	{
		return __pdf();
	}

	static weight_type backwardWeight(const codomain_type sample)
	{
		return backwardPdf(sample);
	}

	template<typename U = vector<T, 1> >
	static quotient_and_pdf<U, T> quotientAndPdf()
	{
		return quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), __pdf());
	}
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct UniformSphere
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
		T z = T(1.0) - T(2.0) * _sample.x;
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
		return vector_t2((T(1.0) - _sample.z) * T(0.5), phi / twopi);
	}

	static domain_type generateInverse(const codomain_type _sample)
	{
		return __generateInverse(_sample);
	}

	static T __pdf()
	{
		return T(1.0) / (T(4.0) * numbers::pi<T>);
	}

	static density_type forwardPdf(const cache_type cache)
	{
		return __pdf();
	}

	static weight_type forwardWeight(const cache_type cache)
	{
		return __pdf();
	}

	static density_type backwardPdf(const vector_t3 _sample)
	{
		return __pdf();
	}

	static weight_type backwardWeight(const codomain_type sample)
	{
		return backwardPdf(sample);
	}

	template<typename U = vector<T, 1> >
	static quotient_and_pdf<U, T> quotientAndPdf()
	{
		return quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), __pdf());
	}
};
} // namespace sampling

} // namespace hlsl
} // namespace nbl

#endif
