// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_PROJECTED_SPHERICAL_RECTANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_PROJECTED_SPHERICAL_RECTANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/sampling/bilinear.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// "Practical Warps" projected solid angle sampler for spherical rectangles.
//
// How it works:
//   1. Build a bilinear patch from NdotL at each of the 4 corner directions
//   2. Warp uniform [0,1]^2 through the bilinear to importance-sample NdotL
//   3. Feed the warped UV into the solid angle sampler to get a rect offset
//   4. PDF = (1/SolidAngle) * bilinearPdf
//
// Template parameter `UsePdfAsWeight`: when true (default), forwardWeight/backwardWeight
// return the PDF instead of the projected-solid-angle MIS weight.
// TODO: the projected-solid-angle MIS weight (UsePdfAsWeight=false) has been shown to be
// poor in practice. Once confirmed by testing, remove the false path and stop storing
// receiverNormal, receiverWasBSDF, and rcpProjSolidAngle as members.
template<typename T, bool UsePdfAsWeight = true>
struct ProjectedSphericalRectangle
{
	using scalar_type = T;
	using vector2_type = vector<T, 2>;
	using vector3_type = vector<T, 3>;
	using vector4_type = vector<T, 4>;

	// BackwardTractableSampler concept types
	using domain_type = vector2_type;
	using codomain_type = vector3_type;
	using density_type = scalar_type;
	using weight_type = density_type;

	struct cache_type
	{
		scalar_type abs_cos_theta;
		vector2_type warped;
		typename Bilinear<scalar_type>::cache_type bilinearCache;
	};

	// NOTE: produces a degenerate (all-zero) bilinear patch when the receiver normal faces away
	// from all four rectangle vertices, resulting in NaN PDFs (0 * inf). Callers must ensure
	// at least one vertex has positive projection onto the receiver normal.
	static ProjectedSphericalRectangle<T,UsePdfAsWeight> create(NBL_CONST_REF_ARG(shapes::SphericalRectangle<T>) shape, const vector3_type observer, const vector3_type _receiverNormal, const bool _receiverWasBSDF)
	{
		ProjectedSphericalRectangle<T,UsePdfAsWeight> retval;
		const vector3_type n = hlsl::mul(shape.basis, _receiverNormal);
		retval.localReceiverNormal = n;
		retval.receiverWasBSDF = _receiverWasBSDF;

		// Compute solid angle and get r0 in local frame (before z-flip)
		const typename shapes::SphericalRectangle<T>::solid_angle_type sa = shape.solidAngle(observer);
		const vector3_type r0 = sa.r0;

		// All 4 corners share r0.z; x is r0.x or r0.x+ex, y is r0.y or r0.y+ey
		const scalar_type r1x = r0.x + shape.extents.x;
		const scalar_type r1y = r0.y + shape.extents.y;

		// Unnormalized dots: dot(corner_i, n)
		const scalar_type base_dot = hlsl::dot(r0, n);
		const scalar_type dx = shape.extents.x * n.x;
		const scalar_type dy = shape.extents.y * n.y;
		const vector4_type dots = vector4_type(base_dot, base_dot + dx, base_dot + dy, base_dot + dx + dy);

		// Squared lengths of each corner
		const scalar_type r0zSq = r0.z * r0.z;
		const vector4_type lenSq = vector4_type(
			r0.x * r0.x + r0.y * r0.y,
			r1x * r1x + r0.y * r0.y,
			r0.x * r0.x + r1y * r1y,
			r1x * r1x + r1y * r1y
		) + hlsl::promote<vector4_type>(r0zSq);

		// dot(normalize(corner), n) = dot(corner, n) * rsqrt(lenSq)
		// Bilinear corners: [0]=v00 [1]=v10 [2]=v01 [3]=v11
		const scalar_type minimumProjSolidAngle = 0.0;
		const vector4_type bxdfPdfAtVertex = math::conditionalAbsOrMax(_receiverWasBSDF,
			dots * hlsl::rsqrt<vector4_type>(lenSq),
			hlsl::promote<vector4_type>(minimumProjSolidAngle));
		retval.bilinearPatch = Bilinear<scalar_type>::create(bxdfPdfAtVertex);

		// Reuse the already-computed solid_angle_type to avoid recomputing mul(basis, origin - observer)
		retval.sphrect = SphericalRectangle<T>::create(sa, shape.extents);
		retval.rcpSolidAngle = retval.sphrect.solidAngle > scalar_type(0.0) ? scalar_type(1.0) / retval.sphrect.solidAngle : scalar_type(0.0);

		NBL_IF_CONSTEXPR(!UsePdfAsWeight)
		{
			const scalar_type projSA = shape.projectedSolidAngleFromLocal(r0, n);
			retval.rcpProjSolidAngle = projSA > scalar_type(0.0) ? scalar_type(1.0) / projSA : scalar_type(0.0);
		}
		return retval;
	}

	// returns a normalized 3D direction in the local frame
	codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		Bilinear<scalar_type> bilinear = bilinearPatch;
		cache.warped = bilinear.generate(u, cache.bilinearCache);
		typename SphericalRectangle<scalar_type>::cache_type sphrectCache;
		const vector3_type dir = sphrect.generate(cache.warped, sphrectCache);
		cache.abs_cos_theta = bilinear.forwardWeight(u, cache.bilinearCache);
		return dir;
	}

	// returns a 2D offset on the rectangle surface from the r0 corner
	vector2_type generateSurfaceOffset(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
	{
		Bilinear<scalar_type> bilinear = bilinearPatch;
		cache.warped = bilinear.generate(u, cache.bilinearCache);
		typename SphericalRectangle<scalar_type>::cache_type sphrectCache;
		const vector2_type sampleOffset = sphrect.generateSurfaceOffset(cache.warped, sphrectCache);
		cache.abs_cos_theta = bilinear.forwardWeight(u, cache.bilinearCache);
		return sampleOffset;
	}

	density_type forwardPdf(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
	{
		return rcpSolidAngle * bilinearPatch.forwardPdf(u, cache.bilinearCache);
	}

	weight_type forwardWeight(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
	{
		if (UsePdfAsWeight)
			return forwardPdf(u, cache);
		return cache.abs_cos_theta * rcpProjSolidAngle;
	}

	// `p` is the normalized [0,1]^2 position on the rectangle
	density_type backwardPdf(const vector2_type p) NBL_CONST_MEMBER_FUNC
	{
		return rcpSolidAngle * bilinearPatch.backwardPdf(p);
	}

	weight_type backwardWeight(const vector2_type p) NBL_CONST_MEMBER_FUNC
	{
		NBL_IF_CONSTEXPR(UsePdfAsWeight)
			return backwardPdf(p);
		const scalar_type minimumProjSolidAngle = 0.0;
		// Reconstruct local direction from normalized rect position
		const vector3_type localDir = hlsl::normalize(sphrect.r0 + vector3_type(
			p.x * sphrect.extents.x,
			p.y * sphrect.extents.y,
			scalar_type(0)
		));
		const scalar_type abs_cos_theta = math::conditionalAbsOrMax(receiverWasBSDF, hlsl::dot(localReceiverNormal, localDir), minimumProjSolidAngle);
		return abs_cos_theta * rcpProjSolidAngle;
	}

	sampling::SphericalRectangle<T> sphrect;
	Bilinear<scalar_type> bilinearPatch;
	scalar_type rcpSolidAngle;
	scalar_type rcpProjSolidAngle;
	vector3_type localReceiverNormal;
	bool receiverWasBSDF;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
