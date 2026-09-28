// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_LIGHTCUT_TREE_WEIGHT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_LIGHTCUT_TREE_WEIGHT_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/sampling/lightcut_tree/node.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// 4-ary stochastic light-cut tree sampler (Estevez-Kulla 2018, simplified): a discrete sampler over
// leaf indices, importance-weighted per cluster by power * orientation (* 1/dist^2 in mode 0). Shading
// position + normal are captured at create() so generate() consumes one random number.
template<typename T, uint32_t Mode>
struct LightcutTreeChildWeight
{
   static T compute(NBL_CONST_REF_ARG(LightcutTreeChild<T>) c, const vector<T, 3> x, const vector<T, 3> n)
   {
      if (!(c.power > T(0)))
         return T(0);

      // Mode is a compile-time template argument, so DXC folds these tests and DCEs the unused branches.
      if (Mode == 2u) // uniform over live children
         return T(1);
      if (Mode == 1u) // power only
         return c.power;

      // Modes 0 and 3 both need the orientation cone bound; only mode 0 needs the distance term.
      const vector<T, 3> ext        = c.bboxMax - c.bboxMin;
      const T            halfDiagSq = T(0.25) * hlsl::dot(ext, ext);

      const vector<T, 3> center         = T(0.5) * (c.bboxMin + c.bboxMax);
      const vector<T, 3> dToCentroid    = center - x;
      const T            centroidDistSq = hlsl::dot(dToCentroid, dToCentroid);


      const T distToCentroidSq = hlsl::max(centroidDistSq, halfDiagSq);
      const T dotND            = hlsl::dot(n, dToCentroid);

      const bool fullyFacing = dotND >= T(0) && dotND * dotND >= distToCentroidSq - halfDiagSq;

      // sinAlpha^2 for the bbox angular radius; the floor on distToCentroidSq keeps it <= 1, no
      // clamp needed. Guarded: the fully-facing path never reads it, mode 4 always does.
      T sinAlphaSq = T(0);
      T cosAlpha   = T(1);
      if (!fullyFacing || Mode == 4u)
      {
         sinAlphaSq = halfDiagSq / distToCentroidSq;
         cosAlpha   = sqrt(hlsl::max(T(1) - sinAlphaSq, T(0)));
      }

      T orientFactor = T(1);
      if (!fullyFacing)
      {
         const T rcpDist = rsqrt(distToCentroidSq);
         const T cosPhi  = dotND * rcpDist;
         const T sinPhi  = sqrt(hlsl::max(T(1) - cosPhi * cosPhi, T(0)));
         orientFactor    = hlsl::max(cosPhi * cosAlpha + sinPhi * sqrt(sinAlphaSq), T(0));
      }
      if (!(orientFactor > T(0)))
         return T(0);

      if (Mode == 3u)
      {
         // Orientation only, NO distance: distance lives in the RIS resample target (numerator).
         return c.power * orientFactor;
      }

      if (Mode == 4u)
      {
         return c.power * (sinAlphaSq / (T(1) + cosAlpha)) * orientFactor;
      }

      // Mode 0:
      const vector<T, 3> dNear     = hlsl::max<vector<T, 3> >(hlsl::max<vector<T, 3> >(c.bboxMin - x, x - c.bboxMax), promote<vector<T, 3> >(T(0)));
      const T            minDistSq = hlsl::dot(dNear, dNear);
      const T            distSq    = hlsl::max(minDistSq, halfDiagSq);
      return c.power * orientFactor / distSq;
   }
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
