// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CONCENTRIC_MAPPING_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/math/functions.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// Based on: Shirley & Chiu, "A Low Distortion Map Between Disk and Square", 1997
// See also: Clarberg, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD", 2008
// http://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
template<typename T>
struct ConcentricMapping
{
   using scalar_type = T;
   using vector2_type = vector<T, 2>;
   using vector3_type = vector<T, 3>;
   using vector4_type = vector<T, 4>;

   // BijectiveSampler concept types
   using domain_type = vector2_type;
   using codomain_type = vector2_type;
   using density_type = scalar_type;
   using weight_type = density_type;

   struct cache_type
   {
      // TODO: should we cache `r`?
   };

   static codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
   {
      // map [0,1]^2 to [-1,1]^2
      const vector2_type centered = scalar_type(2) * u - hlsl::promote<vector2_type>(scalar_type(1));

      const scalar_type a = centered.x;
      const scalar_type b = centered.y;

      // dominant axis selection
      const bool cond = a * a > b * b;
      const scalar_type dominant = hlsl::select(cond, a, b);
      const scalar_type minor = hlsl::select(cond, b, a);
	  
      const scalar_type safe_dominant = dominant != scalar_type(0) ? dominant : scalar_type(0);
      const scalar_type ratio = minor / safe_dominant;
	  
      const scalar_type angle = scalar_type(0.25) * numbers::pi<scalar_type> * ratio;
      const scalar_type c = hlsl::cos<scalar_type>(angle);
      const scalar_type s = hlsl::sin<scalar_type>(angle);

      // final selection without branching
      const scalar_type x = hlsl::select(cond, c, s);
      const scalar_type y = hlsl::select(cond, s, c);

      return dominant * vector2_type(x, y);
   }

   // Overload for BasicSampler
   static codomain_type generate(domain_type u)
   {
      cache_type dummy;
      return generate(u, dummy);
   }

   static domain_type generateInverse(const codomain_type p)
   {
      const scalar_type r = hlsl::sqrt(p.x * p.x + p.y * p.y);

      const scalar_type ax = hlsl::abs<scalar_type>(p.x);
      const scalar_type ay = hlsl::abs<scalar_type>(p.y);

      // swapped = ay > ax
      const bool swapped = ay > ax;

      // branchless selection
      const scalar_type num = hlsl::select(swapped, ax, ay);
      const scalar_type denom = hlsl::select(swapped, ay, ax);

      // angle in [0, pi/4]
      const scalar_type phi = hlsl::atan2(num, denom);

      const scalar_type minor_val = r * phi / (scalar_type(0.25) * numbers::pi<scalar_type>);

      // reconstruct a,b using select instead of branching
      const scalar_type a_base = hlsl::select(swapped, minor_val, r);
      const scalar_type b_base = hlsl::select(swapped, r, minor_val);

      const scalar_type a = ieee754::copySign(a_base, p.x);
      const scalar_type b = ieee754::copySign(b_base, p.y);

      return (vector2_type(a, b) + hlsl::promote<vector2_type>(scalar_type(1))) * scalar_type(0.5);
   }

   // The PDF of Shirley mapping is constant (1/PI on the unit disk)
   static density_type forwardPdf(const domain_type u, cache_type cache) { return numbers::inv_pi<scalar_type>; }
   static density_type backwardPdf(codomain_type v) { return numbers::inv_pi<scalar_type>; }

   static weight_type forwardWeight(const domain_type u, cache_type cache) { return forwardPdf(u, cache); }
   static weight_type backwardWeight(codomain_type v) { return backwardPdf(v); }
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
