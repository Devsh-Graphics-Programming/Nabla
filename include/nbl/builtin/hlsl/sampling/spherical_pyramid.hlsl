// Copyright (C) 2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_PYRAMID_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_PYRAMID_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/shapes/obb.hlsl>
#include <nbl/builtin/hlsl/shapes/obb_silhouette.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_rectangle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// Tag-dispatched inner sampler factory: overload selected by the type of the
// default-constructed `tag` arg. Avoids the per-inner adapter struct.
inline SphericalRectangle<float32_t> buildInner(float32_t3x3 basis, float32_t2 r0, float32_t2 ext, SphericalRectangle<float32_t> /*tag*/)
{
   return SphericalRectangle<float32_t>::create(basis, float32_t3(r0, 1.0f), ext);
}

inline ProjectedSphericalRectangle<float32_t> buildInner(float32_t3x3 basis, float32_t2 r0, float32_t2 ext, ProjectedSphericalRectangle<float32_t> /*tag*/)
{
   shapes::CompressedSphericalRectangle<float32_t> compressed;
   compressed.origin = basis[0] * r0.x + basis[1] * r0.y + basis[2];
   compressed.right  = basis[0] * ext.x;
   compressed.up     = basis[1] * ext.y;
   return ProjectedSphericalRectangle<float32_t>::create(compressed, float32_t3(0.0f, 0.0f, 0.0f), float32_t3(0.0f, 0.0f, 1.0f), false);
}

// Spherical Pyramid: gnomonic bounding rectangle for silhouette sampling.
//
// UseCaliper=false: axis1 picks the longest world-space silhouette edge
//   (one compare per edge, no inner loop, blind to perpendicular spread).
// UseCaliper=true: spherical rotating-caliper. For each candidate edge (A, B),
//   the extremal opposing vertex C is found via argmax_K dot(C_K, precross)
//   where precross = cross(B-A, n0); this matches argmax dot(n0, cross(C+A, C+B))
//   by the cyclic scalar triple product. Score = cos(dihedral) between the
//   AB-great-circle and the Lexell-circle plane through (-A, -B, C). The
//   lune cosine is a heuristic; the post-search bound pass is exact regardless.
//
// Pipeline: axis3 = normalize(-unnormCentroid); axis1 = project bestEdge3d
// onto plane(axis3); axis2 = cross(axis3, axis1); computeBound3D yields
// (rectR0, rectExtents). axis3 is not stored, reconstructed via getAxis3().
//
// rectR0/rectExtents are returned out-params from createFromVertices and not
// stored on the pyramid (the inner sampler keeps its own copy). The local
// vertex array dies at end-of-create-scope; only the inner sampler persists.
template<bool UseCaliper, typename InnerSampler>
struct SphericalPyramid
{
   using scalar_type   = float32_t;
   using vector2_type  = float32_t2;
   using vector3_type  = float32_t3;
   using domain_type   = vector2_type;
   using codomain_type = vector3_type;
   using density_type  = scalar_type;
   using weight_type   = density_type;

   // Caches the inner sampler's cache plus a pre-computed `pdf` that bakes in
   // the silhouette/horizon validity test from generate().
   struct cache_type
   {
      typename InnerSampler::cache_type inner;
      density_type                      pdf;
   };

   float32_t3 axis1;
   float32_t3 axis2; // axis3 reconstructed via getAxis3() = cross(axis1, axis2)

   // Per-edge cross products in world space. Populated during Pass 1's
   // centroid accumulation (also cached for caliper scoring), used by
   // isInside(dir) in generate().
   shapes::SilEdgeNormals silEdgeNormals;

   // Constructed by create(silhouette, view) via tag-dispatched buildInner.
   // The synth-vertices path (createFromVertices direct) leaves it default-init.
   InnerSampler inner;

   float32_t3 getAxis3() NBL_CONST_MEMBER_FUNC { return cross(axis1, axis2); }

   // Pass 1: per-edge cross + Stokes centroid; UseCaliper=false also tracks
   // the longest world edge here. Out params exist in both modes so the
   // per-count cascade has one signature; DCE drops the longest-edge body when
   // UseCaliper=true.
   template<uint32_t I, uint32_t J>
   void processEdge(float32_t3 vertices[shapes::MaxOBBSilhouetteVertices], NBL_REF_ARG(float32_t3) unnormCentroid, NBL_REF_ARG(float32_t) bestLenSq, NBL_REF_ARG(float32_t3) bestEdge3d, NBL_REF_ARG(uint32_t) bestEdge)
   {
      const float32_t3 vI = vertices[I];
      const float32_t3 vJ = vertices[J];

      const float32_t3 c            = cross(vI, vJ);
      silEdgeNormals.edgeNormals[I] = c;
      unnormCentroid += c;

      if (!UseCaliper)
      {
         // Explicit nbl::hlsl::select so DXC emits scalar-conditional OpSelect
         // for the vec3 update instead of a bool-broadcast v3bool.
         const float32_t3 edge3d = vJ - vI;
         const float32_t  lenSq  = dot(edge3d, edge3d);
         const bool       isBest = lenSq > bestLenSq;
         bestLenSq               = max(lenSq, bestLenSq);
         bestEdge3d              = nbl::hlsl::select(isBest, edge3d, bestEdge3d);
         bestEdge                = nbl::hlsl::select(isBest, I, bestEdge);
      }
   }

   // Caliper-only helpers (DCE'd when UseCaliper=false).

   // Track the silhouette vertex with max dot(vK, precross). SkipA/SkipB are
   // the candidate edge's (I, J); compile-time skipped (drops the verts[K]
   // read entirely). Assumes vertices are ~unit length so we can skip the
   // per-K |vK| factor in the cosine.
   template<uint32_t K, uint32_t SkipA, uint32_t SkipB>
   static void tryK(float32_t3 vertices[shapes::MaxOBBSilhouetteVertices], float32_t3 precross, NBL_REF_ARG(float32_t) bestNum, NBL_REF_ARG(float32_t3) bestC)
   {
      if (K != SkipA && K != SkipB)
      {
         const float32_t3 vK     = vertices[K];
         const float32_t  num    = dot(vK, precross);
         const bool       better = num > bestNum;
         bestNum                 = max(num, bestNum);
         bestC                   = nbl::hlsl::select(better, vK, bestC);
      }
   }

   // Cascade-on-count K scan with (I, J) as compile-time skips. bestNum seeds
   // at -inf; bestC's placeholder is always overwritten (count >= 3).
   template<uint32_t I, uint32_t J>
   static float32_t3 findExtremalC(float32_t3 vertices[shapes::MaxOBBSilhouetteVertices], uint32_t count, float32_t3 precross)
   {
      float32_t  bestNum = -1e30f;
      float32_t3 bestC   = vertices[0];
      tryK<0, I, J>(vertices, precross, bestNum, bestC);
      tryK<1, I, J>(vertices, precross, bestNum, bestC);
      tryK<2, I, J>(vertices, precross, bestNum, bestC);
      if (count > 3)
      {
         tryK<3, I, J>(vertices, precross, bestNum, bestC);
         if (count > 4)
         {
            tryK<4, I, J>(vertices, precross, bestNum, bestC);
            if (count > 5)
            {
               tryK<5, I, J>(vertices, precross, bestNum, bestC);
               if (count > 6)
                  tryK<6, I, J>(vertices, precross, bestNum, bestC);
            }
         }
      }
      return bestC;
   }

   // Score candidate edge (I, J) by cos(dihedral) between AB-great-circle
   // and Lexell plane through (-A, -B, C_win). Identity used:
   //   cross(C+A, C+B) = n0 + cross(A, C) + cross(C, B)
   // so we reuse cached n0. Larger score = smaller bounding lune. max(.,1e-30f)
   // keeps rsqrt finite on collapsed edges (they lose on numerator anyway).
   template<uint32_t I, uint32_t J>
   static void evalCandidate(float32_t3 vertices[shapes::MaxOBBSilhouetteVertices], uint32_t count, NBL_CONST_REF_ARG(shapes::SilEdgeNormals) sen, NBL_REF_ARG(float32_t) bestScore, NBL_REF_ARG(float32_t3) bestEdge3d, NBL_REF_ARG(uint32_t) bestEdge)
   {
      const float32_t3 vI     = vertices[I];
      const float32_t3 vJ     = vertices[J];
      const float32_t3 n0     = sen.edgeNormals[I];
      const float32_t3 edge3d = vJ - vI;

      const float32_t3 precross = cross(edge3d, n0);
      const float32_t3 C        = findExtremalC<I, J>(vertices, count, precross);

      const float32_t3 lexell_n1   = n0 + cross(vI, C) + cross(C, vJ);
      const float32_t  numerator   = dot(n0, lexell_n1);
      const float32_t  edgeDenomSq = dot(n0, n0) * dot(lexell_n1, lexell_n1);
      const float32_t  score       = numerator * rsqrt(max(edgeDenomSq, 1e-30f));

      const bool better = score > bestScore;
      bestScore         = max(score, bestScore);
      bestEdge3d        = nbl::hlsl::select(better, edge3d, bestEdge3d);
      bestEdge          = nbl::hlsl::select(better, I, bestEdge);
   }

   // Gnomonic-project each silhouette vertex into the (axis1, axis2, axis3)
   // frame and accumulate the AABB.
   template<uint32_t I>
   static void boundOne3D(float32_t3 vertices[shapes::MaxOBBSilhouetteVertices], float32_t3 axis1, float32_t3 perp, float32_t3 axis3, NBL_REF_ARG(float32_t4) bound)
   {
      const float32_t3 vert  = vertices[I];
      const float32_t  rcpDp = rcp(dot(vert, axis3));
      const float32_t  x     = dot(vert, axis1) * rcpDp;
      const float32_t  y     = dot(vert, perp) * rcpDp;
      bound.x                = min(bound.x, x);
      bound.y                = min(bound.y, y);
      bound.z                = max(bound.z, x);
      bound.w                = max(bound.w, y);
   }

   static void computeBound3D(float32_t3 vertices[shapes::MaxOBBSilhouetteVertices], uint32_t count, float32_t3 axis1, float32_t3 perp, float32_t3 axis3, NBL_REF_ARG(float32_t4) bound)
   {
      bound = float32_t4(1e10f, 1e10f, -1e10f, -1e10f);
      boundOne3D<0>(vertices, axis1, perp, axis3, bound);
      boundOne3D<1>(vertices, axis1, perp, axis3, bound);
      boundOne3D<2>(vertices, axis1, perp, axis3, bound);
      if (count > 3)
      {
         boundOne3D<3>(vertices, axis1, perp, axis3, bound);
         if (count > 4)
         {
            boundOne3D<4>(vertices, axis1, perp, axis3, bound);
            if (count > 5)
            {
               boundOne3D<5>(vertices, axis1, perp, axis3, bound);
               if (count > 6)
                  boundOne3D<6>(vertices, axis1, perp, axis3, bound);
            }
         }
      }
   }

   // Pyramid from pre-materialized verts; (rectR0, rectExtents) returned as
   // out-params (not stored on the pyramid).
   static SphericalPyramid<UseCaliper, InnerSampler> createFromVertices(float32_t3 vertices[shapes::MaxOBBSilhouetteVertices], uint32_t count, NBL_REF_ARG(float32_t2) outRectR0, NBL_REF_ARG(float32_t2) outRectExtents)
   {
      SphericalPyramid<UseCaliper, InnerSampler> self;
      // Sentinel-init so unused slots (count..6) produce dot(dir,(0,0,-1)) < 0
      // for the sign-bit AND in shapes::SilEdgeNormals::isInside.
      self.silEdgeNormals = shapes::SilEdgeNormals::initSentinel();

      // Tiny z-bias seed so symmetric shapes don't normalize(0) to NaN; the
      // cross sum dominates for any non-degenerate silhouette.
      // verts past count are zero-init by materialize, so reading them is harmless.
      float32_t3 unnormCentroid = float32_t3(0.0f, 0.0f, 1e-6f);
      float32_t  bestLenSq      = 0.0f;
      float32_t3 bestEdge3d     = float32_t3(1.0f, 0.0f, 0.0f);
      uint32_t   bestEdge       = 0;

      self.processEdge<0, 1>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
      self.processEdge<1, 2>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
      if (count == 3)
      {
         self.processEdge<2, 0>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
      }
      else
      {
         self.processEdge<2, 3>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
         if (count == 4)
         {
            self.processEdge<3, 0>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
         }
         else
         {
            self.processEdge<3, 4>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
            if (count == 5)
            {
               self.processEdge<4, 0>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
            }
            else
            {
               self.processEdge<4, 5>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
               if (count == 6)
               {
                  self.processEdge<5, 0>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
               }
               else // count == 7
               {
                  self.processEdge<5, 6>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
                  self.processEdge<6, 0>(vertices, unnormCentroid, bestLenSq, bestEdge3d, bestEdge);
               }
            }
         }
      }

      const float32_t3 axis3 = normalize(-unnormCentroid);

      // Pass 2: caliper dihedral scan overwrites bestEdge3d. Skipped under
      // UseCaliper=false (keeps Pass 1's longest edge).
      if (UseCaliper)
      {
         float32_t bestScore = -2.0f;

         evalCandidate<0, 1>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
         evalCandidate<1, 2>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
         if (count == 3)
         {
            evalCandidate<2, 0>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
         }
         else
         {
            evalCandidate<2, 3>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
            if (count == 4)
            {
               evalCandidate<3, 0>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
            }
            else
            {
               evalCandidate<3, 4>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
               if (count == 5)
               {
                  evalCandidate<4, 0>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
               }
               else
               {
                  evalCandidate<4, 5>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
                  if (count == 6)
                  {
                     evalCandidate<5, 0>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
                  }
                  else // count == 7
                  {
                     evalCandidate<5, 6>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
                     evalCandidate<6, 0>(vertices, count, self.silEdgeNormals, bestScore, bestEdge3d, bestEdge);
                  }
               }
            }
         }
      }

      // axis1 = winning chord projected onto plane(axis3) and normalized.
      // max(lenSq, 1e-12) keeps rsqrt finite; degenerate select picks a stable
      // axis perpendicular to axis3.
      const float32_t3 inPlaneEdge  = bestEdge3d - axis3 * dot(bestEdge3d, axis3);
      const float32_t  inPlaneLenSq = dot(inPlaneEdge, inPlaneEdge);
      const bool       useY         = abs(axis3.x) >= 0.9f;
      const float32_t  scale        = rsqrt(max(inPlaneLenSq, 1e-12f));

      const bool       degenerate    = inPlaneLenSq <= 1e-12f;
      const float32_t3 fallbackAxis1 = nbl::hlsl::select(useY, float32_t3(0.0f, 1.0f, 0.0f), float32_t3(1.0f, 0.0f, 0.0f));
      self.axis1                     = nbl::hlsl::select(degenerate, fallbackAxis1, inPlaneEdge * scale);
      self.axis2                     = cross(axis3, self.axis1);

      float32_t4 bestBound;
      computeBound3D(vertices, count, self.axis1, self.axis2, axis3, bestBound);

      // Per-axis degenerate clamp: each upper bound at least 1e-6 above lower.
      // Independent per axis so a single collapsed axis doesn't kill the other.
      bestBound.zw = max(bestBound.zw, bestBound.xy + 1e-6f);

      outRectR0      = bestBound.xy;
      outRectExtents = float32_t2(bestBound.zw - bestBound.xy);

      // Pre-rotate edge normals into local frame so per-sample inside test
      // can use the cheaper 2D form (2 muls + 2 adds + n.z per edge instead
      // of 3 muls + 2 adds). Amortized once per build; saves 7 muls/sample.
      self.silEdgeNormals.transformToLocal(self.axis1, self.axis2, axis3);

      return self;
   }

   // Materialize verts (in shading-point-relative coords baked into silhouette)
   // from the silhouette, build the pyramid, then construct the InnerSampler
   // via tag-dispatched buildInner. Local rect data dies at end-of-scope; only
   // the inner sampler retains a copy.
   static SphericalPyramid<UseCaliper, InnerSampler> create(NBL_CONST_REF_ARG(shapes::ClippedSilhouette) silhouette, shapes::OBBView<float32_t> view)
   {
      float32_t3 vertices[shapes::MaxOBBSilhouetteVertices];
      silhouette.materialize(view, vertices);

      float32_t2 rectR0, rectExtents;
      SphericalPyramid<UseCaliper, InnerSampler> self = createFromVertices(vertices, silhouette.count, rectR0, rectExtents);

      // tag's value is unread; only its type selects the overload.
      const float32_t3x3 basis = float32_t3x3(self.axis1, self.axis2, self.getAxis3());
      InnerSampler tag;
      self.inner = buildInner(basis, rectR0, rectExtents, tag);
      return self;
   }

   // Generate via inner.generateNormalizedLocal so we can recover gnomonic
   // (localX, localY) for the 2D inside test. With rectR0.z == 1, localDir.z =
   // 1/hitDist, so localDir.{x,y} * hitDist == gnomonic coords. Bake
   // silhouette/horizon validity into cache.pdf so forwardPdf is O(1).
   codomain_type generate(domain_type u, NBL_REF_ARG(cache_type) cache)
   {
      scalar_type          hitDist;
      const codomain_type  localDir = inner.generateNormalizedLocal(u, cache.inner, hitDist);
      const codomain_type  dir      = localDir.x * axis1 + localDir.y * axis2 + localDir.z * getAxis3();
      const scalar_type    localX   = localDir.x * hitDist;
      const scalar_type    localY   = localDir.y * hitDist;
      const bool           valid    = dir.z > 0.0f && silEdgeNormals.isInsideLocal(localX, localY);
      cache.pdf                     = hlsl::select(valid, inner.forwardPdf(u, cache.inner), 0.0f);
      return dir;
   }

   density_type forwardPdf(domain_type u, cache_type cache) NBL_CONST_MEMBER_FUNC { return cache.pdf; }
   weight_type  forwardWeight(domain_type u, cache_type cache) NBL_CONST_MEMBER_FUNC { return cache.pdf; }
   uint32_t     selectedIdx(cache_type cache) NBL_CONST_MEMBER_FUNC { return 0u; }
};

}
}
}

#endif
