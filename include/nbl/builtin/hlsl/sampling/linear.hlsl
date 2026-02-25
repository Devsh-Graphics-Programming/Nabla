// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_LINEAR_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_LINEAR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/sampling/warp_and_pdf.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct Linear
{
	using scalar_type = T;
	using vector2_type = vector<T, 2>;

	// BijectiveSampler concept types
	using domain_type = scalar_type;
	using codomain_type = scalar_type;
	using density_type = scalar_type;
	using sample_type = codomain_and_rcpPdf<codomain_type, density_type>;
	using inverse_sample_type = domain_and_rcpPdf<domain_type, density_type>;

	static Linear<T> create(const vector2_type linearCoeffs) // start and end importance values (start, end)
	{
		Linear<T> retval;
		retval.linearCoeffStart = linearCoeffs[0];
		retval.rcpDiff = 1.0 / (linearCoeffs[0] - linearCoeffs[1]);
		vector2_type squaredCoeffs = linearCoeffs * linearCoeffs;
		retval.squaredCoeffStart = squaredCoeffs[0];
		retval.squaredCoeffDiff = squaredCoeffs[1] - squaredCoeffs[0];
		return retval;
	}

	scalar_type generate(const scalar_type u)
	{
		return hlsl::mix(u, (linearCoeffStart - hlsl::sqrt(squaredCoeffStart + u * squaredCoeffDiff)) * rcpDiff, hlsl::abs(rcpDiff) < numeric_limits<scalar_type>::max);
	}

	scalar_type linearCoeffStart;
	scalar_type rcpDiff;
	scalar_type squaredCoeffStart;
	scalar_type squaredCoeffDiff;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
