// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/math/quaternions.hlsl>
#include <nbl/builtin/hlsl/math/fast_acos.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/ieee754.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

enum SphericalTriangleAlgorithm : uint16_t
{
   STA_ARVO = 0,
   STA_PBRT = 1
};

namespace impl
{

// DifferenceOfProducts: a*b - c*d with Kahan FMA compensation
template<typename T>
T differenceOfProducts(T a, T b, T c, T d)
{
   const T cd = c * d;
   const T dop = nbl::hlsl::fma(a, b, -cd);
   const T err = nbl::hlsl::fma(-c, d, cd);
   return dop + err;
}

// SumOfProducts: a*b + c*d with Kahan FMA compensation
template<typename T>
T sumOfProducts(T a, T b, T c, T d)
{
   const T cd = c * d;
   const T sop = nbl::hlsl::fma(a, b, cd);
   const T err = nbl::hlsl::fma(c, d, -cd);
   return sop + err;
}

} // namespace impl

template<typename T, SphericalTriangleAlgorithm Algorithm = STA_ARVO>
struct SphericalTriangle
{
   using scalar_type = T;
   using vector2_type = vector<T, 2>;
   using vector3_type = vector<T, 3>;

   using domain_type = vector2_type;
   using codomain_type = vector3_type;
   using density_type = scalar_type;
   using weight_type = density_type;

   struct cache_type
   {
   };

   static SphericalTriangle create(NBL_CONST_REF_ARG(shapes::SphericalTriangle<T>) tri)
   {
      SphericalTriangle retval;
      retval.rcpSolidAngle = scalar_type(1.0) / tri.solid_angle;
      retval.tri_vertices[0] = tri.vertices[0];
      retval.tri_vertices[1] = tri.vertices[1];
      retval.triCosc = tri.cos_sides[2];
      // precompute great circle normal of arc AC: cross(A,C) has magnitude sin(b),
      // so multiplying by csc(b) normalizes it; zero when side AC is degenerate
      const scalar_type cscb = tri.csc_sides[1];
      const vector3_type arcACPlaneNormal = hlsl::cross(tri.vertices[0], tri.vertices[2]) * hlsl::select(cscb < numeric_limits<scalar_type>::max, cscb, scalar_type(0));
      retval.e_C = hlsl::cross(arcACPlaneNormal, tri.vertices[0]);
      retval.cosA = tri.cos_vertices[0];
      retval.sinA = tri.sin_vertices[0];
      if (Algorithm == STA_ARVO)
      {
         retval.sinA_triCosc = retval.sinA * retval.triCosc;
         retval.eCdotB = hlsl::dot(retval.e_C, tri.vertices[1]);
      }
      retval.APlusC = tri.vertices[0] + tri.vertices[2];
      return retval;
   }

   codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
   {
      // Step 1: compute sin/cos of A_hat and the angle difference (A_hat - alpha)
      const scalar_type A_hat = u.x / rcpSolidAngle;
      scalar_type sinA_hat, cosA_hat;
      math::sincos(A_hat, sinA_hat, cosA_hat);
      const scalar_type s = sinA_hat * cosA - cosA_hat * sinA; // sin(A_hat - alpha)
      const scalar_type t = cosA_hat * cosA + sinA_hat * sinA; // cos(A_hat - alpha)

      // Step 2: compute cos(b') and sin(b') -- arc from A to the new vertex C'
      scalar_type cosBp, sinBp;
      if (Algorithm == STA_ARVO) // faster than PBRT
      {
         const scalar_type u_ = t - cosA;
         const scalar_type v_ = s + sinA_triCosc;
         const scalar_type num = (v_ * t - u_ * s) * cosA - v_;
         const scalar_type denum = (v_ * s + u_ * t) * sinA;

#define ACCURATE 1
#if ACCURATE
         // sqrt(1 - cosBp^2) loses precision when cosBp ~ 1 (small u.x).
         // Use stable factorization: sinBp = sqrt((denum-num)(denum+num)) / |denum|
         // where denum-num = sinA*(1+triCosc)*(1-cosA_hat).

         // For large triangles with high u.x, cosA_hat can approach -1,
         // making (1 + cosA_hat) near zero and the division unstable.
         // Use the algebraic identity only when cosA_hat > 0 (safe denominator).
         const scalar_type rcpDenum = scalar_type(1) / denum;
         const scalar_type oneMinusCosAhat = hlsl::select(cosA_hat > scalar_type(0), sinA_hat * sinA_hat / (scalar_type(1) + cosA_hat), scalar_type(1) - cosA_hat);
         const scalar_type DminusN = sinA * (scalar_type(1) + triCosc) * oneMinusCosAhat;
         sinBp = sqrt<scalar_type>(max<scalar_type>(scalar_type(0), DminusN * (denum + num))) * nbl::hlsl::abs(rcpDenum);
         cosBp = scalar_type(1) - DminusN * rcpDenum;
#else // 17% faster, less accurate
         cosBp = num / denum;
         sinBp = sqrt<scalar_type>(max<scalar_type>(scalar_type(0), scalar_type(1) - cosBp * cosBp));
#endif
      }
      else // STA_PBRT, accurate, slowest
      {
         // PBRT uses cosPhi = -t, sinPhi = -s (pi offset from Arvo's A_hat)
         const scalar_type k1 = -t + cosA;
         const scalar_type k2 = -s - sinA * triCosc;
         cosBp = (k2 + impl::differenceOfProducts(k2, -t, k1, -s) * cosA) / (impl::sumOfProducts(k2, -s, k1, -t) * sinA);
         cosBp = nbl::hlsl::clamp(cosBp, scalar_type(-1), scalar_type(1));
         sinBp = sqrt<scalar_type>(scalar_type(1) - cosBp * cosBp);
      }

      // Step 3: construct C' on the great circle through A toward C
      const vector3_type cp = cosBp * tri_vertices[0] + sinBp * e_C;

      // Step 4: uniformly sample the great circle arc from B to C'
      scalar_type cosCpB;
      NBL_IF_CONSTEXPR(Algorithm == STA_ARVO)
         cosCpB = cosBp * triCosc + sinBp * eCdotB;
      else
         cosCpB = nbl::hlsl::dot(cp, tri_vertices[1]);
      // TODO: degeneracy at u.y = 0. z = 1 - u.y*(1-cosCpB) makes sinZ = sqrt(1-z^2) behave like
      // sqrt(u.y) near zero, so dL/du.y diverges as u.y^(-1/2) and every higher derivative diverges
      // faster. The forward Jacobian test in 37_HLSLSamplingTests reports ~2-8% error at u.y < 0.003
      // even with the O(h^2) one-sided stencil because the third-derivative term dominates. At
      // u.y = 0 exactly, L collapses to vertex B for all u.x (|det J| = 0), so it's an intrinsic
      // property of the Arvo parameterization, not a bug. Fix: rework the arc interpolation to use
      // a u.y -> angle mapping whose derivatives stay bounded near u.y = 0 (e.g. acos(z) = angle
      // from B, then sample arc-length linearly), so the Jacobian is smooth and the skip band in
      // the tester can be removed.
      const scalar_type z = scalar_type(1) - u.y * (scalar_type(1) - cosCpB);
      const scalar_type sinZ = sqrt<scalar_type>(max<scalar_type>(scalar_type(0), scalar_type(1) - z * z));
      return z * tri_vertices[1] + sinZ * hlsl::normalize(cp - cosCpB * tri_vertices[1]);
   }

   // generate() works in two steps:
   //   u.x -> pick C' on arc AC (choosing a sub-area fraction)
   //   u.y -> pick L on arc B->C' (linear interpolation)
   //
   // So the inverse is:
   //   1) Find C': intersect great circle (B,L) with great circle (A,C)
   //   2) u.x = solidAngle(A,B,C') / solidAngle(A,B,C)
   //   3) u.y = |L-B|^2 / |C'-B|^2
   domain_type generateInverse(const codomain_type L) NBL_CONST_MEMBER_FUNC
   {
      // Step 1: find C' = intersection of great circles (B,L) and (A,C)
      const vector3_type BxL = nbl::hlsl::cross(tri_vertices[1], L);
      const scalar_type sinBL_sq = nbl::hlsl::dot(BxL, BxL);

      // C' lies on arc AC, so C' = A*cos(t) + e_C*sin(t).
      // C' also lies on the B-L plane, so dot(BxL, C') = 0.
      // Solving: (cos(t), sin(t)) = (tripleE, -tripleA) / R
      const scalar_type tripleA = nbl::hlsl::dot(BxL, tri_vertices[0]);
      const scalar_type tripleE = nbl::hlsl::dot(BxL, e_C);
      const scalar_type R_sq = tripleA * tripleA + tripleE * tripleE;

      if (sinBL_sq < numeric_limits<scalar_type>::epsilon || R_sq < numeric_limits<scalar_type>::epsilon)
      {
         // Recover u.y from |L-B|^2 / |A-B|^2 (using C'=A; the (1-cosCpB) ratio
         // cancels so any C' gives the same result).
         const vector3_type LminusB = L - tri_vertices[1];
         const vector3_type AminusB = tri_vertices[0] - tri_vertices[1];
         const scalar_type v_num = nbl::hlsl::dot(LminusB, LminusB);
         const scalar_type v_denom = nbl::hlsl::dot(AminusB, AminusB);
         const scalar_type v = hlsl::select(v_denom > numeric_limits<scalar_type>::epsilon,
            nbl::hlsl::clamp(v_num / v_denom, scalar_type(0.0), scalar_type(1.0)),
            scalar_type(0.0));
         return vector2_type(scalar_type(0.0), v);
      }

      const scalar_type rcpR = scalar_type(1.0) / nbl::hlsl::sqrt(R_sq);
      vector3_type cp = tri_vertices[0] * (tripleE * rcpR) + e_C * (-tripleA * rcpR);
      // two intersections exist; pick the one on the minor arc A->C (branchless sign flip)
      cp = ieee754::flipSignIfRHSNegative(cp, nbl::hlsl::dot(cp, APlusC));

      // Step 2: u.x = solidAngle(A,B,C') / solidAngle(A,B,C)
      // Van Oosterom-Strackee: tan(Omega/2) = |A.(BxC')| / (1 + A.B + B.C' + C'.A)
      //
      // Numerator stability: the naive triple product dot(A, cross(B, C')) suffers
      // catastrophic cancellation when C' is near A (small u.x), because
      // cross(B, C') ~ cross(B, A) and dot(A, cross(B, A)) = 0 exactly.
      // Expanding C' = cosBp*A + sinBp*e_C into the triple product:
      //   A.(BxC') = cosBp * A.(BxA) + sinBp * A.(Bxe_C) = sinBp * A.(Bxe_C)
      // since A.(BxA) = 0 identically. This avoids the cancellation.
      const scalar_type cosBp_inv = nbl::hlsl::dot(cp, tri_vertices[0]);
      const scalar_type sinBp_inv = nbl::hlsl::dot(cp, e_C);
      const scalar_type AxBdotE = nbl::hlsl::dot(tri_vertices[0], nbl::hlsl::cross(tri_vertices[1], e_C));
      const scalar_type num = sinBp_inv * AxBdotE;
      const scalar_type cosCpB = nbl::hlsl::dot(tri_vertices[1], cp);
      const scalar_type den = scalar_type(1.0) + triCosc + cosCpB + cosBp_inv;
      const scalar_type subSolidAngle = scalar_type(2.0) * nbl::hlsl::atan2(nbl::hlsl::abs(num), den);
      const scalar_type u = nbl::hlsl::clamp(subSolidAngle * rcpSolidAngle, scalar_type(0.0), scalar_type(1.0));

      // Step 3: u.y = |L-B|^2 / |C'-B|^2
      // Squared Euclidean distance avoids catastrophic cancellation vs (1-dot)/(1-dot)
      const vector3_type LminusB = L - tri_vertices[1];
      const vector3_type cpMinusB = cp - tri_vertices[1];
      const scalar_type v_num = nbl::hlsl::dot(LminusB, LminusB);
      const scalar_type v_denom = nbl::hlsl::dot(cpMinusB, cpMinusB);
      const scalar_type v = hlsl::select(v_denom > numeric_limits<scalar_type>::epsilon,
         nbl::hlsl::clamp(v_num / nbl::hlsl::max(v_denom, numeric_limits<scalar_type>::min),
            scalar_type(0.0), scalar_type(1.0)),
         scalar_type(0.0));

      return vector2_type(u, v);
   }

   density_type forwardPdf(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
   {
      return rcpSolidAngle;
   }

   weight_type forwardWeight(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
   {
      return forwardPdf(u, cache);
   }

   density_type backwardPdf(const codomain_type L) NBL_CONST_MEMBER_FUNC
   {
      return rcpSolidAngle;
   }

   weight_type backwardWeight(const codomain_type L) NBL_CONST_MEMBER_FUNC
   {
      return backwardPdf(L);
   }

   scalar_type rcpSolidAngle;
   scalar_type cosA;
   scalar_type sinA;
   scalar_type sinA_triCosc; // precomputed sinA * triCosc
   scalar_type eCdotB; // precomputed dot(e_C, tri_vertices[1]), Arvo only

   vector3_type tri_vertices[2]; // A and B only
   scalar_type triCosc;
   vector3_type e_C; // precomputed cross(arcACPlaneNormal, A), unit vector perp to A in A-C plane
   vector3_type APlusC; // precomputed A + C, used to pick the minor-arc intersection in generateInverse
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
