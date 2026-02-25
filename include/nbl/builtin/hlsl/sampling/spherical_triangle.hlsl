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
#include <nbl/builtin/hlsl/sampling/warp_and_pdf.hlsl>

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

	// BijectiveSampler concept types
	using domain_type = vector2_type;
	using codomain_type = vector3_type;
	using density_type = scalar_type;
	using sample_type = codomain_and_rcpPdf<codomain_type, density_type>;
	using inverse_sample_type = domain_and_rcpPdf<domain_type, density_type>;

	static SphericalTriangle<T> create(NBL_CONST_REF_ARG(shapes::SphericalTriangle<T>) tri)
	{
		SphericalTriangle<T> retval;
		retval.tri = tri;
		return retval;
	}

	// WARNING: can and will return NAN if one or three of the triangle edges are near zero length
	vector3_type generate(scalar_type solidAngle, const vector3_type cos_vertices, const vector3_type sin_vertices, scalar_type cos_a, scalar_type cos_c, scalar_type csc_b, scalar_type csc_c, const vector2_type u)
	{
		scalar_type negSinSubSolidAngle, negCosSubSolidAngle;
		math::sincos(solidAngle * u.x - numbers::pi<scalar_type>, negSinSubSolidAngle, negCosSubSolidAngle);

		const scalar_type p = negCosSubSolidAngle * sin_vertices[0] - negSinSubSolidAngle * cos_vertices[0];
		const scalar_type q = -negSinSubSolidAngle * sin_vertices[0] - negCosSubSolidAngle * cos_vertices[0];

		// TODO: we could optimize everything up and including to the first slerp, because precision here is just godawful
		scalar_type u_ = q - cos_vertices[0];
		scalar_type v_ = p + sin_vertices[0] * cos_c;

		// the slerps could probably be optimized by sidestepping `normalize` calls and accumulating scaling factors
		vector3_type C_s = tri.vertex0;
		if (csc_b < numeric_limits<scalar_type>::max)
		{
			const scalar_type cosAngleAlongAC = ((v_ * q - u_ * p) * cos_vertices[0] - v_) / ((v_ * p + u_ * q) * sin_vertices[0]);
			if (nbl::hlsl::abs(cosAngleAlongAC) < 1.f)
				C_s += math::quaternion<scalar_type>::slerp_delta(tri.vertex0, tri.vertex2 * csc_b, cosAngleAlongAC);
		}

		vector3_type retval = tri.vertex1;
		const scalar_type cosBC_s = nbl::hlsl::dot(C_s, tri.vertex1);
		const scalar_type csc_b_s = 1.0 / nbl::hlsl::sqrt(1.0 - cosBC_s * cosBC_s);
		if (csc_b_s < numeric_limits<scalar_type>::max)
		{
			const scalar_type cosAngleAlongBC_s = nbl::hlsl::clamp(1.0 + cosBC_s * u.y - u.y, -1.f, 1.f);
			if (nbl::hlsl::abs(cosAngleAlongBC_s) < 1.f)
				retval += math::quaternion<scalar_type>::slerp_delta(tri.vertex1, C_s * csc_b_s, cosAngleAlongBC_s);
		}
		return retval;
	}

	vector3_type generate(NBL_REF_ARG(scalar_type) rcpPdf, const vector2_type u)
	{
		scalar_type cos_a, cos_c, csc_b, csc_c;
		vector3_type cos_vertices, sin_vertices;

		rcpPdf = tri.solidAngleOfTriangle(cos_vertices, sin_vertices, cos_a, cos_c, csc_b, csc_c);

		return generate(rcpPdf, cos_vertices, sin_vertices, cos_a, cos_c, csc_b, csc_c, u);
	}

	vector2_type generateInverse(NBL_REF_ARG(scalar_type) pdf, scalar_type solidAngle, const vector3_type cos_vertices, const vector3_type sin_vertices, scalar_type cos_a, scalar_type cos_c, scalar_type csc_b, scalar_type csc_c, const vector3_type L)
	{
		pdf = 1.0 / solidAngle;

		const scalar_type cosAngleAlongBC_s = nbl::hlsl::dot(L, tri.vertex1);
		const scalar_type csc_a_ = 1.0 / nbl::hlsl::sqrt(1.0 - cosAngleAlongBC_s * cosAngleAlongBC_s);
		const scalar_type cos_b_ = nbl::hlsl::dot(L, tri.vertex0);

		const scalar_type cosB_ = (cos_b_ - cosAngleAlongBC_s * cos_c) * csc_a_ * csc_c;
		const scalar_type sinB_ = nbl::hlsl::sqrt(1.0 - cosB_ * cosB_);

		const scalar_type cosC_ = sin_vertices[0] * sinB_ * cos_c - cos_vertices[0] * cosB_;
		const scalar_type sinC_ = nbl::hlsl::sqrt(1.0 - cosC_ * cosC_);

		math::sincos_accumulator<scalar_type> angle_adder = math::sincos_accumulator<scalar_type>::create(cos_vertices[0], sin_vertices[0]);
		angle_adder.addAngle(cosB_, sinB_);
		angle_adder.addAngle(cosC_, sinC_);
		const scalar_type subTriSolidAngleRatio = (angle_adder.getSumofArccos() - numbers::pi<scalar_type>)*pdf;
		const scalar_type u = subTriSolidAngleRatio > numeric_limits<scalar_type>::min ? subTriSolidAngleRatio : 0.0;

		const scalar_type cosBC_s = (cos_vertices[0] + cosB_ * cosC_) / (sinB_ * sinC_);
		const scalar_type v = (1.0 - cosAngleAlongBC_s) / (1.0 - (cosBC_s < bit_cast<float>(0x3f7fffff) ? cosBC_s : cos_c));

		return vector2_type(u, v);
	}

	vector2_type generateInverse(NBL_REF_ARG(scalar_type) pdf, const vector3_type L)
	{
		scalar_type cos_a, cos_c, csc_b, csc_c;
		vector3_type cos_vertices, sin_vertices;

		const scalar_type solidAngle = tri.solidAngleOfTriangle(cos_vertices, sin_vertices, cos_a, cos_c, csc_b, csc_c);

		return generateInverse(pdf, solidAngle, cos_vertices, sin_vertices, cos_a, cos_c, csc_b, csc_c, L);
	}

	shapes::SphericalTriangle<T> tri;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
