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
template<typename T, bool UsePdfAsWeight = true>
struct ProjectedSphericalRectangle
{
   using scalar_type  = T;
   using vector2_type = vector<T, 2>;
   using vector3_type = vector<T, 3>;
   using vector4_type = vector<T, 4>;

   // BackwardTractableSampler concept types
   using domain_type   = vector2_type;
   using codomain_type = vector3_type;
   using density_type  = scalar_type;
   using weight_type   = density_type;

   struct cache_type
   {
      typename Bilinear<scalar_type>::cache_type bilinearCache;
      vector3_type L; // TODO: same as projected triangle w.r.t. UsePdfAsWeight==false
   };

   // Shared finalization for both create() overloads: builds the bilinear patch, the inner sphrect
   // sampler, and the UsePdfAsWeight=false extras. The two overloads differ only in how they
   // compute bxdfPdfAtVertex (worldspace corner normalizations vs local-frame rsqrt(lenSq)).
   static ProjectedSphericalRectangle<T, UsePdfAsWeight> create(NBL_CONST_REF_ARG(shapes::SphericalRectangle<T>) shape, NBL_CONST_REF_ARG(typename shapes::SphericalRectangle<T>::solid_angle_type) sa,
      const vector4_type bxdfPdfAtVertex, const vector3_type _receiverNormal)
   {
      ProjectedSphericalRectangle<T, UsePdfAsWeight> retval;
      retval.bilinearPatch = Bilinear<scalar_type>::create(bxdfPdfAtVertex);
      // Reuse solid_angle_type to avoid recomputing mul(basis, origin - observer)
      retval.sphrect = SphericalRectangle<T>::create(shape.basis, sa, shape.extents);
      NBL_IF_CONSTEXPR(!UsePdfAsWeight)
      {
         retval.receiverNormal     = _receiverNormal;
         const vector3_type nLocal = hlsl::mul(shape.basis, _receiverNormal);
         retval.projSolidAngle     = shape.projectedSolidAngleFromLocal(sa.r0, nLocal);
      }
      return retval;
   }

   // Shouldn't produce NAN if all corners have 0 proj solid angle due to min density adds/clamps in the linear sampler
   static ProjectedSphericalRectangle<T, UsePdfAsWeight> create(NBL_CONST_REF_ARG(shapes::CompressedSphericalRectangle<T>) compressed, const vector3_type observer, const vector3_type _receiverNormal, const bool _receiverWasBSDF)
   {
      // 4 normalized worldspace corners dotted with the worldspace receiver normal. Avoids the
      // mul(basis, receiverNormal) data dependency of the uncompressed overload so these 4
      // normalize+dot chains can pipeline alongside the basis/solid-angle work below.
      const vector3_type c0   = compressed.origin - observer;
      const vector3_type c1   = c0 + compressed.right;
      const vector3_type c2   = c0 + compressed.up;
      const vector3_type c3   = c1 + compressed.up;
      const vector4_type dots = vector4_type(
         hlsl::dot(hlsl::normalize(c0), _receiverNormal),
         hlsl::dot(hlsl::normalize(c1), _receiverNormal),
         hlsl::dot(hlsl::normalize(c2), _receiverNormal),
         hlsl::dot(hlsl::normalize(c3), _receiverNormal));
      const scalar_type minimumProjSolidAngle = scalar_type(0.0);
      const vector4_type bxdfPdfAtVertex      = math::conditionalAbsOrMax(_receiverWasBSDF, dots, hlsl::promote<vector4_type>(minimumProjSolidAngle));

      const shapes::SphericalRectangle<T> shape                         = shapes::SphericalRectangle<T>::create(compressed);
      const typename shapes::SphericalRectangle<T>::solid_angle_type sa = shape.solidAngle(observer);
      return create(shape, sa, bxdfPdfAtVertex, _receiverNormal);
   }

   // Shouldn't produce NAN if all corners have 0 proj solid angle due to min density adds/clamps in the linear sampler
   static ProjectedSphericalRectangle<T, UsePdfAsWeight> create(NBL_CONST_REF_ARG(shapes::SphericalRectangle<T>) shape, const vector3_type observer, const vector3_type _receiverNormal, const bool _receiverWasBSDF)
   {
      // Local-frame path: unnormalized dot(corner_i, n) with n = basis * receiverNormal, then
      // a single rsqrt<vec4>(lenSq) for all 4 corner normalizations at once.
      const vector3_type n                                              = hlsl::mul(shape.basis, _receiverNormal);
      const typename shapes::SphericalRectangle<T>::solid_angle_type sa = shape.solidAngle(observer);
      const vector3_type r0                                             = sa.r0;

      // All 4 corners share r0.z; x is r0.x or r0.x+ex, y is r0.y or r0.y+ey
      const scalar_type r1x      = r0.x + shape.extents.x;
      const scalar_type r1y      = r0.y + shape.extents.y;
      const scalar_type base_dot = hlsl::dot(r0, n);
      const scalar_type dx       = shape.extents.x * n.x;
      const scalar_type dy       = shape.extents.y * n.y;
      const vector4_type dots    = vector4_type(base_dot, base_dot + dx, base_dot + dy, base_dot + dx + dy);

      const scalar_type r0zSq  = r0.z * r0.z;
      const vector4_type lenSq = vector4_type(
                                    r0.x * r0.x + r0.y * r0.y,
                                    r1x * r1x + r0.y * r0.y,
                                    r0.x * r0.x + r1y * r1y,
                                    r1x * r1x + r1y * r1y) +
         hlsl::promote<vector4_type>(r0zSq);

      // dot(normalize(corner), n) = dot(corner, n) * rsqrt(lenSq). Bilinear corners: [0]=v00 [1]=v10 [2]=v01 [3]=v11
      const scalar_type minimumProjSolidAngle = scalar_type(0.0);
      const vector4_type bxdfPdfAtVertex      = math::conditionalAbsOrMax(_receiverWasBSDF,
         dots * hlsl::rsqrt<vector4_type>(lenSq),
         hlsl::promote<vector4_type>(minimumProjSolidAngle));

      return create(shape, sa, bxdfPdfAtVertex, _receiverNormal);
   }

   // returns a normalized 3D direction in the local frame
   codomain_type generateNormalizedLocal(const domain_type u, NBL_REF_ARG(cache_type) cache, NBL_REF_ARG(scalar_type) hitDist) NBL_CONST_MEMBER_FUNC
   {
      const vector2_type warped = bilinearPatch.generate(u, cache.bilinearCache);
      typename SphericalRectangle<scalar_type>::cache_type sphrectCache; // there's nothing in the cache
      const vector3_type dir = sphrect.generateNormalizedLocal(warped, sphrectCache, hitDist);
      NBL_IF_CONSTEXPR(!UsePdfAsWeight)
      cache.L = hlsl::mul(hlsl::transpose(sphrect.basis), dir);
      return dir;
   }

   // returns a unnormalized 3D direction in the global frame
   codomain_type generateUnnormalized(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
   {
      const vector2_type warped = bilinearPatch.generate(u, cache.bilinearCache);
      typename SphericalRectangle<scalar_type>::cache_type sphrectCache; // there's nothing in the cache
      const vector3_type dir = sphrect.generateUnnormalized(warped, sphrectCache);
      NBL_IF_CONSTEXPR(!UsePdfAsWeight)
      cache.L = dir * hlsl::rsqrt(hlsl::dot(dir, dir));
      return dir;
   }

   // returns a normalized 3D direction in the global frame
   codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
   {
      const vector2_type warped = bilinearPatch.generate(u, cache.bilinearCache);
      typename SphericalRectangle<scalar_type>::cache_type sphrectCache; // there's nothing in the cache
      const vector3_type dir = sphrect.generate(warped, sphrectCache);
      NBL_IF_CONSTEXPR(!UsePdfAsWeight)
      cache.L = dir;
      return dir;
   }

   density_type forwardPdf(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
   {
      return bilinearPatch.forwardPdf(u, cache.bilinearCache) / sphrect.solidAngle;
   }

   weight_type forwardWeight(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
   {
      NBL_IF_CONSTEXPR(UsePdfAsWeight)
      return forwardPdf(u, cache);
      return backwardWeight(cache.L);
   }

   weight_type backwardWeight(const codomain_type L) NBL_CONST_MEMBER_FUNC
   {
      NBL_IF_CONSTEXPR(UsePdfAsWeight)
      {
#if 0
			const vector2_type warped = sphrect.generateInvese(L); // TODO: implement `generateInverse`
			return bilinearPatch.backwardPdf(warped) / sphrect.solidAngle;
#endif
         return hlsl::numeric_limits<weight_type>::quiet_NaN;
      }
      // make the MIS weight always abs because even when receiver is a BRDF, the samples in lower hemisphere will get killed and MIS weight never used
      return hlsl::abs(hlsl::dot(L, receiverNormal)) / projSolidAngle;
   }

   sampling::SphericalRectangle<T> sphrect;
   Bilinear<scalar_type> bilinearPatch;
   // TODO: same as projected triangle w.r.t. UsePdfAsWeight==false
   vector3_type receiverNormal;
   scalar_type projSolidAngle;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
