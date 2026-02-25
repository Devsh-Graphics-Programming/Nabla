// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/sampling/bilinear.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/sampling/quotient_and_pdf.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T>
struct ProjectedSphericalTriangle
{
	using scalar_type = T;
	using vector2_type = vector<T, 2>;
	using vector3_type = vector<T, 3>;
	using vector4_type = vector<T, 4>;

	// BackwardDensitySampler concept types
	using domain_type = vector2_type;
	using codomain_type = vector3_type;
	using density_type = scalar_type;
	using sample_type = codomain_and_rcpPdf<codomain_type, density_type>;

	static ProjectedSphericalTriangle<T> create(NBL_CONST_REF_ARG(shapes::SphericalTriangle<T>) tri)
	{
		ProjectedSphericalTriangle<T> retval;
		retval.tri = tri;
		return retval;
	}

	vector4_type computeBilinearPatch(const vector3_type receiverNormal, bool isBSDF)
	{
		const scalar_type minimumProjSolidAngle = 0.0;

		matrix<T, 3, 3> m = matrix<T, 3, 3>(tri.vertex0, tri.vertex1, tri.vertex2);
		const vector3_type bxdfPdfAtVertex = math::conditionalAbsOrMax(isBSDF, nbl::hlsl::mul(m, receiverNormal), hlsl::promote<vector3_type>(minimumProjSolidAngle));

		return bxdfPdfAtVertex.yyxz;
	}

	vector3_type generate(NBL_REF_ARG(scalar_type) rcpPdf, scalar_type solidAngle, const vector3_type cos_vertices, const vector3_type sin_vertices, scalar_type cos_a, scalar_type cos_c, scalar_type csc_b, scalar_type csc_c, const vector3_type receiverNormal, bool isBSDF, const vector2_type _u)
	{
		vector2_type u;
		// pre-warp according to proj solid angle approximation
		vector4_type patch = computeBilinearPatch(receiverNormal, isBSDF);
		Bilinear<scalar_type> bilinear = Bilinear<scalar_type>::create(patch);
		u = bilinear.generate(rcpPdf, _u);

		// now warp the points onto a spherical triangle
		const vector3_type L = sphtri.generate(solidAngle, cos_vertices, sin_vertices, cos_a, cos_c, csc_b, csc_c, u);
		rcpPdf *= solidAngle;

		return L;
	}

	vector3_type generate(NBL_REF_ARG(scalar_type) rcpPdf, const vector3_type receiverNormal, bool isBSDF, const vector2_type u)
	{
		scalar_type cos_a, cos_c, csc_b, csc_c;
		vector3_type cos_vertices, sin_vertices;
		const scalar_type solidAngle = tri.solidAngleOfTriangle(cos_vertices, sin_vertices, cos_a, cos_c, csc_b, csc_c);
		return generate(rcpPdf, solidAngle, cos_vertices, sin_vertices, cos_a, cos_c, csc_b, csc_c, receiverNormal, isBSDF, u);
	}

	scalar_type pdf(scalar_type solidAngle, const vector3_type cos_vertices, const vector3_type sin_vertices, scalar_type cos_a, scalar_type cos_c, scalar_type csc_b, scalar_type csc_c, const vector3_type receiverNormal, bool receiverWasBSDF, const vector3_type L)
	{
		scalar_type pdf;
		const vector2_type u = sphtri.generateInverse(pdf, solidAngle, cos_vertices, sin_vertices, cos_a, cos_c, csc_b, csc_c, L);

		vector4_type patch = computeBilinearPatch(receiverNormal, receiverWasBSDF);
		Bilinear<scalar_type> bilinear = Bilinear<scalar_type>::create(patch);
		return pdf * bilinear.pdf(u);
	}

	scalar_type pdf(const vector3_type receiverNormal, bool receiverWasBSDF, const vector3_type L)
	{
		scalar_type pdf;
		const vector2_type u = sphtri.generateInverse(pdf, L);

		vector4_type patch = computeBilinearPatch(receiverNormal, receiverWasBSDF);
		Bilinear<scalar_type> bilinear = Bilinear<scalar_type>::create(patch);
		return pdf * bilinear.pdf(u);
	}

	shapes::SphericalTriangle<T> tri;
	sampling::SphericalTriangle<T> sphtri;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
