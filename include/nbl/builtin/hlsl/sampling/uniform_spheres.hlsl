// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_SPHERES_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_UNIFORM_SPHERES_INCLUDED_

#include "nbl/builtin/hlsl/concepts.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/sampling/quotient_and_pdf.hlsl"
#include "nbl/builtin/hlsl/sampling/warp_and_pdf.hlsl"

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
	using sample_type = codomain_and_rcpPdf<codomain_type, density_type>;
	using inverse_sample_type = domain_and_rcpPdf<domain_type, density_type>;

	static vector_t3 __generate(const vector_t2 _sample)
	{
		T z = _sample.x;
		T r = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(1.0) - z * z));
		T phi = T(2.0) * numbers::pi<T> * _sample.y;
		return vector_t3(r * hlsl::cos<T>(phi), r * hlsl::sin<T>(phi), z);
	}

	vector_t3 generate(const vector_t2 _sample)
	{
		return __generate(_sample);
	}

	static vector_t2 __generateInverse(const vector_t3 _sample)
	{
		T phi = hlsl::atan2(_sample.y, _sample.x);
		const T twopi = T(2.0) * numbers::pi<T>;
		phi += hlsl::mix(T(0.0), twopi, phi < T(0.0));
		return vector_t2(_sample.z, phi / twopi);
	}

	vector_t2 generateInverse(const vector_t3 _sample)
	{
		return __generateInverse(_sample);
	}

	static scalar_type __pdf()
	{
		return T(1.0) / (T(2.0) * numbers::pi<T>);
	}

	scalar_type forwardPdf(const vector_t2 _sample)
	{
		return __pdf();
	}

	scalar_type backwardPdf(const vector_t3 _sample)
	{
		return __pdf();
	}

	template<typename U = vector<T, 1> >
	static ::nbl::hlsl::sampling::quotient_and_pdf<U, T> quotient_and_pdf()
	{
		return ::nbl::hlsl::sampling::quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), pdf());
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
	using sample_type = codomain_and_rcpPdf<codomain_type, density_type>;
	using inverse_sample_type = domain_and_rcpPdf<domain_type, density_type>;

	static vector_t3 __generate(const vector_t2 _sample)
	{
		T z = T(1.0) - T(2.0) * _sample.x;
		T r = hlsl::sqrt<T>(hlsl::max<T>(T(0.0), T(1.0) - z * z));
		T phi = T(2.0) * numbers::pi<T> * _sample.y;
		return vector_t3(r * hlsl::cos<T>(phi), r * hlsl::sin<T>(phi), z);
	}

	vector_t3 generate(const vector_t2 _sample)
	{
		return __generate(_sample);
	}

	static vector_t2 __generateInverse(const vector_t3 _sample)
	{
		T phi = hlsl::atan2(_sample.y, _sample.x);
		const T twopi = T(2.0) * numbers::pi<T>;
		phi += hlsl::mix(T(0.0), twopi, phi < T(0.0));
		return vector_t2((T(1.0) - _sample.z) * T(0.5), phi / twopi);
	}

	vector_t2 generateInverse(const vector_t3 _sample)
	{
		return __generateInverse(_sample);
	}

	static T __pdf()
	{
		return T(1.0) / (T(4.0) * numbers::pi<T>);
	}

	scalar_type forwardPdf(const vector_t2 _sample)
	{
		return __pdf();
	}

	scalar_type backwardPdf(const vector_t3 _sample)
	{
		return __pdf();
	}

	template<typename U = vector<T, 1> >
	static ::nbl::hlsl::sampling::quotient_and_pdf<U, T> quotient_and_pdf()
	{
		return ::nbl::hlsl::sampling::quotient_and_pdf<U, T>::create(hlsl::promote<U>(1.0), pdf());
	}
};
} // namespace sampling

} // namespace hlsl
} // namespace nbl

#endif
