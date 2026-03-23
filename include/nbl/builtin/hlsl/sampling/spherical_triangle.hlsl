// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/quaternions.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct SphericalTriangle
{
	using scalar_type = T;
	using vector2_type = vector<T, 2>;
	using vector3_type = vector<T, 3>;
	using uint_type = unsigned_integer_of_size_t<sizeof(scalar_type)>;

	// BijectiveSampler concept types
	using domain_type = vector2_type;
	using codomain_type = vector3_type;
	using density_type = scalar_type;
	using weight_type = density_type;

	struct cache_type {};

	static SphericalTriangle<T> create(NBL_CONST_REF_ARG(shapes::SphericalTriangle<T>) tri)
	{
		SphericalTriangle<T> retval;
		retval.rcpSolidAngle = scalar_type(1.0) / tri.solid_angle;
		retval.cosA = tri.cos_vertices[0];
		retval.sinA = tri.sin_vertices[0];
		retval.tri_vertices[0] = tri.vertices[0];
		retval.tri_vertices[1] = tri.vertices[1];
		retval.tri_vertices[2] = tri.vertices[2];
		retval.triCosC = tri.cos_sides[2];
		retval.triCscB = tri.csc_sides[1];
		retval.triCscC = tri.csc_sides[2];
		return retval;
	}

	codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
	{
		scalar_type negSinSubSolidAngle, negCosSubSolidAngle;
		const scalar_type solidAngle = scalar_type(1.0) / rcpSolidAngle;
		math::sincos(solidAngle * u.x - numbers::pi<scalar_type>, negSinSubSolidAngle, negCosSubSolidAngle);

		const scalar_type p = negCosSubSolidAngle * sinA - negSinSubSolidAngle * cosA;
		const scalar_type q = -negSinSubSolidAngle * sinA - negCosSubSolidAngle * cosA;

		// TODO: we could optimize everything up and including to the first slerp, because precision here is just godawful
		scalar_type u_ = q - cosA;
		scalar_type v_ = p + sinA * triCosC;

		// the slerps could probably be optimized by sidestepping `normalize` calls and accumulating scaling factors
		vector3_type C_s = tri_vertices[0];
		if (triCscB < numeric_limits<scalar_type>::max)
		{
			const scalar_type cosAngleAlongAC = ((v_ * q - u_ * p) * cosA - v_) / ((v_ * p + u_ * q) * sinA);
			if (nbl::hlsl::abs(cosAngleAlongAC) < 1.f)
				C_s += math::quaternion<scalar_type>::slerp_delta(tri_vertices[0], tri_vertices[2] * triCscB, cosAngleAlongAC);
		}

		vector3_type retval = tri_vertices[1];
		const scalar_type cosBC_s = nbl::hlsl::dot(C_s, tri_vertices[1]);
		const scalar_type csc_b_s = 1.0 / nbl::hlsl::sqrt(1.0 - cosBC_s * cosBC_s);
		if (csc_b_s < numeric_limits<scalar_type>::max)
		{
			const scalar_type cosAngleAlongBC_s = scalar_type(1.0) + cosBC_s * u.y - u.y;
			if (nbl::hlsl::abs(cosAngleAlongBC_s) < scalar_type(1.0))
				retval += math::quaternion<scalar_type>::slerp_delta(tri_vertices[1], C_s * csc_b_s, cosAngleAlongBC_s);
		}

		return retval;
	}

	domain_type generateInverse(const codomain_type L)
	{
		const scalar_type cosAngleAlongBC_s = nbl::hlsl::dot(L, tri_vertices[1]);
		const scalar_type csc_a_ = 1.0 / nbl::hlsl::sqrt(1.0 - cosAngleAlongBC_s * cosAngleAlongBC_s);
		const scalar_type cos_b_ = nbl::hlsl::dot(L, tri_vertices[0]);

		const scalar_type cosB_ = (cos_b_ - cosAngleAlongBC_s * triCosC) * csc_a_ * triCscC;
		const scalar_type sinB_ = nbl::hlsl::sqrt(1.0 - cosB_ * cosB_);

		const scalar_type cosC_ = sinA * sinB_ * triCosC - cosA * cosB_;
		const scalar_type sinC_ = nbl::hlsl::sqrt(1.0 - cosC_ * cosC_);

		math::sincos_accumulator<scalar_type> angle_adder = math::sincos_accumulator<scalar_type>::create(cosA, sinA);
		angle_adder.addAngle(cosB_, sinB_);
		angle_adder.addAngle(cosC_, sinC_);
		const scalar_type subTriSolidAngleRatio = (angle_adder.getSumofArccos() - numbers::pi<scalar_type>) * rcpSolidAngle;
		const scalar_type u = subTriSolidAngleRatio > numeric_limits<scalar_type>::min ? subTriSolidAngleRatio : 0.0;

		const scalar_type sinBC_s_product = sinB_ * sinC_;

		const scalar_type one_below_one = ieee754::nextTowardZero(scalar_type(1.0));
		const scalar_type cosBC_s = sinBC_s_product > numeric_limits<scalar_type>::min ? (cosA + cosB_ * cosC_) / sinBC_s_product : triCosC;
		const scalar_type v_denom = scalar_type(1.0) - (cosBC_s < one_below_one ? cosBC_s : triCosC);
		const scalar_type v = (scalar_type(1.0) - cosAngleAlongBC_s) / v_denom;

		return vector2_type(u, v);
	}

	density_type forwardPdf(const cache_type cache)
	{
		return rcpSolidAngle;
	}

	weight_type forwardWeight(const cache_type cache)
	{
		return forwardPdf(cache);
	}

	density_type backwardPdf(const codomain_type L)
	{
		return rcpSolidAngle;
	}

	weight_type backwardWeight(const codomain_type L)
	{
		return backwardPdf(L);
	}

	scalar_type rcpSolidAngle;
	scalar_type cosA;
	scalar_type sinA;

	vector3_type tri_vertices[3];
	scalar_type triCosC;
	scalar_type triCscB;
	scalar_type triCscC;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
