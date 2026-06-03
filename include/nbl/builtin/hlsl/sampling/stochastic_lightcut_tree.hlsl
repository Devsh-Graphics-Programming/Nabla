// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_STOCHASTIC_LIGHTCUT_TREE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_STOCHASTIC_LIGHTCUT_TREE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl> // ceil / floor / log2 (packed encode)
#include <nbl/builtin/hlsl/bit.hlsl> // bit_cast (packed encode/decode)

// Descent weight mode (what each child's selection weight is):
//   0 = power * orientFactor / dist^2  -- full geometric importance. Unbiased, but the 1/dist^2
//       in the selection pdf (denominator) makes the proposal peaky -> higher variance in
//       1/pProposal. On loose/overlapping clusters the distance estimate isn't accurate enough to
//       pay for that variance, so mode 3 beats it (clean RWMC-ablated A/B). A tighter cluster build
//       could reopen this mode. The unit tests validate this branch.
//   1 = power only -- the descent pdf telescopes to leafPower/totalPower, exactly the alias
//       table's distribution, so power cancels cleanly against contribution and geometry never
//       enters the denominator. Never worse than alias (modulo quantization + tree imbalance).
//   2 = uniform over live children -- ignores power and geometry (A/B floor).
//   3 = power * orientFactor (orientation, NO distance) -- the production renderer descent.
//       Orientation is a bounded [0,1] reweight that can't blow up the denominator, so it stays
//       firefly-/clamp-safe while still culling below-horizon clusters (lower noise than power-
//       only). Distance importance is applied in the RIS resample TARGET (numerator), where a
//       heavy 1/dist^2 only re-ranks a candidate instead of dividing into a clamped spike.
#ifndef NBL_LIGHTCUT_TREE_WEIGHT_MODE
#define NBL_LIGHTCUT_TREE_WEIGHT_MODE 3
#endif

// PDF_FLOOR: stop when the cumulative descent pdf drops below NBL_LIGHTCUT_TREE_PDF_FLOOR.
//            Bounds 1/pdf variance at the leaf level.
// MAX_RATIO: stop when the largest child weight is less than NBL_LIGHTCUT_TREE_STOP_MAX_RATIO of
//            wSum -- i.e. when no child clearly dominates so the next weighted pick is mostly
//            noise (Estevez-Kulla 2018's "no winner" criterion).
#ifndef NBL_LIGHTCUT_TREE_PDF_FLOOR_ENABLED
#define NBL_LIGHTCUT_TREE_PDF_FLOOR_ENABLED 0 // Didn' help on the bench, but left in case a tight SAOH build makes it worth it.
#endif
#ifndef NBL_LIGHTCUT_TREE_PDF_FLOOR
#define NBL_LIGHTCUT_TREE_PDF_FLOOR 1e-1
#endif
#ifndef NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED
#define NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED 1
#endif
#ifndef NBL_LIGHTCUT_TREE_STOP_MAX_RATIO
#define NBL_LIGHTCUT_TREE_STOP_MAX_RATIO 0.2
#endif


namespace nbl
{
namespace hlsl
{
namespace sampling
{

// ----- decoded views (layout-agnostic; what the sampler consumes via NodeAccessor/LeafAccessor, and
// what the packed unpack below produces) -----

NBL_CONSTEXPR_STATIC_INLINE uint32_t LightcutTreeNonEmitterCustomIndex = 0xFFFFFFu;

// One decoded child of a wide-node.
template<typename T>
struct LightcutTreeChild
{
   vector<T, 3> bboxMin;
   vector<T, 3> bboxMax;
   T            power;
};

// Decoded wide-node view: 4 children + per-slot leaf-bit mask.
template<typename T>
struct LightcutTreeWideNode
{
   LightcutTreeChild<T> children[4];
   uint32_t             childLeafMask;
};

// Decoded leaf view (precise bbox, no quantisation).
template<typename T>
struct LightcutTreeLeaf
{
   vector<T, 3> bboxMin;
   vector<T, 3> bboxMax;
   uint32_t     emitterID;
};

// Canonical 32 B CWBVH-4 PACKED representation for StochasticLightcutTreeSampler: the single
// encode/decode contract between the CPU builder and the GPU accessor, so the byte layout lives in one
// place. The sampler stays layout-agnostic (it consumes the decoded LightcutTreeWideNode /
// LightcutTreeLeaf via the NodeAccessor / LeafAccessor concepts); this header is the ready-made packing
// those concepts can be built on.
//
// 32 B wide-node (2 x uint4). 
struct LightcutTreePackedWideNode
{
   float32_t3 origin;
   uint32_t   powExpMask;
   uint32_t4  childPacked;
};

// 32 B leaf: precise fp32 AABB + 32-bit emitter id (fp32 so leaf bboxes don't collapse to fp16 +inf).
struct LightcutTreePackedLeaf
{
   float32_t3 bboxMin;
   float32_t3 bboxMax;
   uint32_t   emitterID;
   uint32_t   _pad;
};

// Sentinel emitterID for padding leaves (no emitter). Decodes back to ~0u.
NBL_CONSTEXPR_STATIC_INLINE uint32_t LightcutTreePackedNoEmitter = 0xFFFFFFFFu;

// ============================================================================
// ----- ENCODE ---------------
// ============================================================================

// Smallest biased exponent (bias 127) b such that 2^(b-127) >= extent. Degenerate axis (extent<=0)
// maps to 0 (scale ~= 0, all quantized values collapse onto the origin).
inline uint32_t lightcutTreePickBiasedExp(const float32_t extent)
{
   if (!(extent > float32_t(0)))
      return 0u;
   const int32_t e = _static_cast<int32_t>(ceil(log2(extent)));
   return _static_cast<uint32_t>(hlsl::clamp(127 + e, 0, 255));
}

// 2^(b-127)/15 via the fp32 bit pattern (bias 127 already matches IEEE-754).
inline float32_t lightcutTreeBiasedExpToScale(const uint32_t biasedExp)
{
   const uint32_t bits = (biasedExp & 0xFFu) << 23u;
   return bit_cast<float32_t>(bits) * (float32_t(1) / float32_t(15));
}

// Quantize one axis to [0,15]. ceilMode: false = floor (for bbox min, conservative low), true = ceil
// (for bbox max, conservative high) -> the decoded child bbox always CONTAINS the true one.
template<bool CeilMode>
uint32_t lightcutTreeQuantize4(const float32_t x, const float32_t invStep)
{
   const float32_t q = CeilMode ? ceil(x * invStep) : floor(x * invStep);
   return _static_cast<uint32_t>(hlsl::clamp(q, float32_t(0), float32_t(15)));
}

inline uint32_t lightcutTreePackRelPower(const float32_t childPower, const float32_t parentPowerSafe)
{
   if (!(childPower > float32_t(0)))
      return 0u;
   const float32_t f = hlsl::clamp(childPower / parentPowerSafe, float32_t(0), float32_t(1));
   return _static_cast<uint32_t>(hlsl::clamp(ceil(f * float32_t(255)), float32_t(1), float32_t(255)));
}

inline uint32_t lightcutTreePackChild(NBL_CONST_REF_ARG(vector<float32_t, 3>) loRel, NBL_CONST_REF_ARG(vector<float32_t, 3>) hiRel, const float32_t scale, const float32_t childPower, const float32_t parentPowerSafe)
{
   const float32_t invStep  = (scale > float32_t(0)) ? (float32_t(1) / scale) : float32_t(0);
   const uint32_t  qLo      = lightcutTreeQuantize4<false>(loRel.x, invStep) | (lightcutTreeQuantize4<false>(loRel.y, invStep) << 4u) | (lightcutTreeQuantize4<false>(loRel.z, invStep) << 8u);
   const uint32_t  qHi      = lightcutTreeQuantize4<true>(hiRel.x, invStep) | (lightcutTreeQuantize4<true>(hiRel.y, invStep) << 4u) | (lightcutTreeQuantize4<true>(hiRel.z, invStep) << 8u);
   const uint32_t  relPower = lightcutTreePackRelPower(childPower, parentPowerSafe);
   return (qLo & 0xFFFu) | ((qHi & 0xFFFu) << 12u) | ((relPower & 0xFFu) << 24u);
}

// Assemble bytes 12-15 from the fp16 parent power, shared exponent, and 4-bit leaf mask.
inline uint32_t lightcutTreePackPowExpMask(const float32_t parentPower, const uint32_t sharedExp, const uint32_t childLeafMask)
{
   const float16_t hp   = _static_cast<float16_t>(hlsl::min(parentPower, float32_t(65504)));
   const uint32_t  bits = _static_cast<uint32_t>(bit_cast<uint16_t>(hp));
   return (bits & 0xFFFFu) | ((sharedExp & 0xFFu) << 16u) | ((childLeafMask & 0xFu) << 24u);
}

// ============================================================================
// -----  DECODE  ----------
// ============================================================================

// Decode bytes 12-15 into the parent power (the per-child scale + leaf mask are read inline by the
// node unpack since they feed the per-child loop).
template<typename T>
T lightcutTreeUnpackParentPower(const uint32_t powExpMask)
{
   return T(bit_cast<float16_t>(uint16_t(powExpMask & 0xFFFFu)));
}

// Full wide-node decode into the sampler's LightcutTreeWideNode<T>. The shared scale broadcasts to
// all 3 axes; childPower = parentPower * relPower/255.
template<typename T>
LightcutTreeWideNode<T> lightcutTreeUnpackWideNode(NBL_CONST_REF_ARG(LightcutTreePackedWideNode) packed)
{
   LightcutTreeWideNode<T> decoded;
   const vector<T, 3>      origin      = vector<T, 3>(packed.origin);
   const T                 parentPower = lightcutTreeUnpackParentPower<T>(packed.powExpMask);
   const T                 scale       = T(lightcutTreeBiasedExpToScale((packed.powExpMask >> 16u) & 0xFFu));
   decoded.childLeafMask               = (packed.powExpMask >> 24u) & 0xFu;

   for (uint32_t s = 0u; s < 4u; ++s)
   {
      const uint32_t     cp       = packed.childPacked[s];
      const uint32_t     qLo      = cp & 0xFFFu;
      const uint32_t     qHi      = (cp >> 12u) & 0xFFFu;
      const uint32_t     powByte  = (cp >> 24u) & 0xFFu;
      const vector<T, 3> qLoF     = vector<T, 3>(T(qLo & 0xFu), T((qLo >> 4u) & 0xFu), T((qLo >> 8u) & 0xFu));
      const vector<T, 3> qHiF     = vector<T, 3>(T(qHi & 0xFu), T((qHi >> 4u) & 0xFu), T((qHi >> 8u) & 0xFu));
      decoded.children[s].bboxMin = origin + qLoF * scale;
      decoded.children[s].bboxMax = origin + qHiF * scale;
      decoded.children[s].power   = parentPower * (T(powByte) * (T(1) / T(255)));
   }
   return decoded;
}

template<typename T>
LightcutTreeLeaf<T> lightcutTreeUnpackLeaf(NBL_CONST_REF_ARG(LightcutTreePackedLeaf) packed)
{
   LightcutTreeLeaf<T> decoded;
   decoded.bboxMin   = vector<T, 3>(packed.bboxMin);
   decoded.bboxMax   = vector<T, 3>(packed.bboxMax);
   decoded.emitterID = hlsl::select(packed.emitterID == LightcutTreePackedNoEmitter, ~uint32_t(0), packed.emitterID);
   return decoded;
}

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

      // Receiver-side cosine upper bound over the whole bbox. phi = angle(n, dirToCentroid);
      // widen by the bbox angular radius alpha (sin(alpha) = halfDiag/distToCentroid) and take
      // cos(max(phi - alpha, 0)). The halfDiag floor on the cone distance also doubles as the
      // proposal heuristic for inside-bounding-sphere queries: it makes orientFactor fall off as
      // sinPhi when the centroid is behind the normal, which is a looser-but-useful proxy for
      // "how much of the cluster lies above the horizon". A tight orient=1 upper bound there
      // collapses top-level descent to power-only in large scenes (regressed FLIP).
      const T distToCentroidSq = hlsl::max(centroidDistSq, halfDiagSq);
      const T dotND            = hlsl::dot(n, dToCentroid);

      // Fully-facing fast path. orientFactor saturates to 1 exactly when cosPhi >= cosAlpha. Cross-
      // multiplying that comparison through distToCentroidSq > 0 (and using sinAlpha^2 = halfDiagSq/
      // distToCentroidSq, whose min(.,1) is inactive because distToCentroidSq >= halfDiagSq) turns it
      // into a transcendental-free test: cosPhi >= 0 (i.e. dotND >= 0) AND dotND^2 >= distToCentroidSq
      // - halfDiagSq. The floor keeps the RHS >= 0; inside the bounding sphere it is 0 so the test
      // collapses to dotND >= 0, matching cosAlpha = 0. Only grazing/partial-cone children pay the
      // rsqrt + sqrt below. cos(phi - alpha) = cosPhi cosAlpha + sinPhi sinAlpha.
      T orientFactor;
      if (dotND >= T(0) && dotND * dotND >= distToCentroidSq - halfDiagSq)
      {
         orientFactor = T(1);
      }
      else
      {
         const T rcpDist  = rsqrt(distToCentroidSq);
         const T cosPhi   = dotND * rcpDist;
         const T sinAlpha = hlsl::min(sqrt(halfDiagSq) * rcpDist, T(1));
         const T cosAlpha = sqrt(hlsl::max(T(1) - sinAlpha * sinAlpha, T(0)));
         const T sinPhi   = sqrt(hlsl::max(T(1) - cosPhi * cosPhi, T(0)));
         orientFactor     = hlsl::max(cosPhi * cosAlpha + sinPhi * sinAlpha, T(0));
      }
      if (!(orientFactor > T(0)))
         return T(0);

      if (Mode == 3u)
      {
         // Orientation only, NO distance: distance lives in the RIS resample target (numerator).
         return c.power * orientFactor;
      }

      // Mode 0:
      const vector<T, 3> dNear     = hlsl::max<vector<T, 3> >(hlsl::max<vector<T, 3> >(c.bboxMin - x, x - c.bboxMax), promote<vector<T, 3> >(T(0)));
      const T            minDistSq = hlsl::dot(dNear, dNear);
      const T            distSq    = hlsl::max(minDistSq, halfDiagSq);
      return c.power * orientFactor / distSq;
   }
};

// No-op SubtreeAliasAccessor for callers that don't enable any early-stop criterion. sample() is
// dead code in that case (gated by #if NBL_LIGHTCUT_TREE_PDF_FLOOR_ENABLED / STOP_MAX_RATIO_ENABLED),
// so this stub is sufficient to satisfy the template signature without paying for the accessor.
template<typename T, typename Codomain>
struct NoSubtreeAliasAccessor
{
   static NoSubtreeAliasAccessor create()
   {
      NoSubtreeAliasAccessor r;
      return r;
   }

   void sample(Codomain W, T u, NBL_REF_ARG(Codomain) outLeafArrayIdx, NBL_REF_ARG(T) outPdf) NBL_CONST_MEMBER_FUNC
   {
      outLeafArrayIdx = Codomain(0);
      outPdf          = T(0);
   }

   // Backward counterpart of sample(), called by backwardPdf's MAX_RATIO mirror. Dead code under
   // the same gating as sample() (only reached when an early-stop criterion is enabled), so the
   // stub returns 0.
   T backwardPdf(Codomain W, Codomain leafArrayIdx) NBL_CONST_MEMBER_FUNC { return T(0); }
};

// Stochastic light-cut tree sampler. Concept-conforming (generate / forwardPdf
// / backwardPdf / cache_type) so it can drop into the same plumbing as
// PackedAliasTable* and other discrete samplers.
//
// SubtreeAliasAccessor concept: two methods
//   void sample(codomain_type W, domain_type u,
//               NBL_REF_ARG(codomain_type) outLeafArrayIdx,
//               NBL_REF_ARG(density_type)  outAliasPdf) NBL_CONST_MEMBER_FUNC
// Picks one leaf in the subtree rooted at internal heap node W via a power-weighted alias table,
// outputs the leaf's LEAF-ARRAY index (caller adds firstLeafIdx for the heap index) and the
// within-subtree leaf-selection pdf. Used by generate()'s early-stop path.
//   density_type backwardPdf(codomain_type W, codomain_type leafArrayIdx) NBL_CONST_MEMBER_FUNC
// The within-subtree pdf that sample(W, .) would assign to the leaf at LEAF-ARRAY index
// leafArrayIdx. Used by backwardPdf()'s MAX_RATIO mirror so forward/backward agree under MIS.
// Both used only when an early-stop criterion (NBL_LIGHTCUT_TREE_PDF_FLOOR / STOP_MAX_RATIO) is
// enabled; when both are disabled the accessor is never called -- a no-op stub is sufficient.
template<typename T, typename Codomain, typename NodeAccessor, typename LeafAccessor, typename SubtreeAliasAccessor, uint32_t Mode NBL_PRIMARY_REQUIRES(concepts::FloatingPointScalar<T>&& concepts::UnsignedIntegralScalar<Codomain>)
struct StochasticLightcutTreeSampler
{
   using scalar_type   = T;
   using domain_type   = T;
   using codomain_type = Codomain; // leaf HEAP index
   using density_type  = scalar_type;
   using weight_type   = density_type;
   using point_type    = vector<T, 3>;
   using wide_node_t   = LightcutTreeWideNode<T>;
   using leaf_t        = LightcutTreeLeaf<T>;

   struct cache_type
   {
      density_type pdf; // pdf of the picked leaf
      leaf_t       leaf; // precise bbox + emitter id, so callers don't re-tap
   };

   static StochasticLightcutTreeSampler create(
      NBL_CONST_REF_ARG(NodeAccessor) _nodeAcc, NBL_CONST_REF_ARG(LeafAccessor) _leafAcc, NBL_CONST_REF_ARG(SubtreeAliasAccessor) _subtreeAcc, const codomain_type _firstLeafIdx, const point_type _shadingPoint, const point_type _shadingNormal)
   {
      StochasticLightcutTreeSampler retval;
      retval.nodeAcc       = _nodeAcc;
      retval.leafAcc       = _leafAcc;
      retval.subtreeAcc    = _subtreeAcc;
      retval.firstLeafIdx  = _firstLeafIdx;
      retval.shadingPoint  = _shadingPoint;
      retval.shadingNormal = _shadingNormal;
      return retval;
   }

   // Cacheless overload (satisfies BasicSampler, mirrors PackedAliasTable::generate(u)): runs the
   // descent and returns just the picked leaf's heap index, discarding the populated cache.
   codomain_type generate(const domain_type u_in) NBL_CONST_MEMBER_FUNC
   {
      cache_type cache;
      return generate(u_in, cache);
   }

   // Walks the heap top-down. One NodeAccessor::get per internal level + one
   // LeafAccessor::get when a leaf is hit. Returns the picked leaf's HEAP
   // index (not its leaf-array index) so MIS callers can climb back up with
   // backwardPdf without remembering firstLeafIdx separately.
   codomain_type generate(const domain_type u_in, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
   {
      cache.pdf = density_type(0);
      // Single-leaf tree: no internal nodes, leaf sits at heap 0.
      if (firstLeafIdx == codomain_type(0))
      {
         leaf_t leaf;
         leafAcc.template get<leaf_t, codomain_type>(codomain_type(0), leaf);
         cache.leaf = leaf;
         cache.pdf  = density_type(1);
         return codomain_type(0);
      }

      codomain_type W   = codomain_type(0);
      density_type  pdf = density_type(1);
      domain_type   xi  = u_in;
      // Descent. Bounded by tree depth in practice; the loop exits via the
      // childIsLeaf return path. The fallback return handles malformed trees
      // where the leaf bit is never set on the chosen child.
      NBL_HLSL_LOOP
      for (uint32_t step = 0u; step < 32u; ++step)
      {
#if NBL_LIGHTCUT_TREE_PDF_FLOOR_ENABLED
         // Cumulative pdf dropped low enough that 1/pdf variance dominates whatever
         // discrimination the next weighted step could add. Delegate to the per-subtree
         // alias table: O(1) power-weighted pick over W's leaves, multiplied into the
         // already-accumulated descent pdf.
         if (step > 0u && pdf < density_type(NBL_LIGHTCUT_TREE_PDF_FLOOR))
         {
            codomain_type aliasLeafArr;
            density_type  aliasPdf;
            subtreeAcc.sample(W, xi, aliasLeafArr, aliasPdf);
            if (!(aliasPdf > density_type(0)))
               return ~codomain_type(0);
            leaf_t leaf;
            leafAcc.template get<leaf_t, codomain_type>(aliasLeafArr, leaf);
            cache.leaf = leaf;
            cache.pdf  = pdf * aliasPdf;
            return firstLeafIdx + aliasLeafArr;
         }
#endif
         wide_node_t w;
         nodeAcc.template get<wide_node_t, codomain_type>(W, w);

         const density_type w0   = LightcutTreeChildWeight<T, Mode>::compute(w.children[0], shadingPoint, shadingNormal);
         const density_type w1   = LightcutTreeChildWeight<T, Mode>::compute(w.children[1], shadingPoint, shadingNormal);
         const density_type w2   = LightcutTreeChildWeight<T, Mode>::compute(w.children[2], shadingPoint, shadingNormal);
         const density_type w3   = LightcutTreeChildWeight<T, Mode>::compute(w.children[3], shadingPoint, shadingNormal);
         const density_type wSum = w0 + w1 + w2 + w3;
         if (!(wSum > density_type(0)))
            return ~codomain_type(0);

#if NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED
         // Estevez-Kulla "no clear winner": the largest child weight is less than the
         // stop-threshold fraction of wSum, so the next weighted pick is mostly noise.
         // Delegate to W's subtree alias instead.
         const density_type wMax = hlsl::max(hlsl::max(w0, w1), hlsl::max(w2, w3));
         if (wMax < density_type(NBL_LIGHTCUT_TREE_STOP_MAX_RATIO) * wSum)
         {
            codomain_type aliasLeafArr;
            density_type  aliasPdf;
            subtreeAcc.sample(W, xi, aliasLeafArr, aliasPdf);
            if (!(aliasPdf > density_type(0)))
               return ~codomain_type(0);
            leaf_t leaf;
            leafAcc.template get<leaf_t, codomain_type>(aliasLeafArr, leaf);
            cache.leaf = leaf;
            cache.pdf  = pdf * aliasPdf;
            return firstLeafIdx + aliasLeafArr;
         }
#endif

         // CDF pick with rescale (branchless).
         const density_type t  = xi * wSum;
         const density_type t1 = t - w0;
         const density_type t2 = t1 - w1;
         const density_type t3 = t2 - w2;
         const bool         m0 = t < w0;
         const bool         m1 = t1 < w1;
         const bool         m2 = t2 < w2;

         const uint32_t     slot  = hlsl::select(m0, 0u, hlsl::select(m1, 1u, hlsl::select(m2, 2u, 3u)));
         const density_type wPick = hlsl::select(m0, w0, hlsl::select(m1, w1, hlsl::select(m2, w2, w3)));
         const density_type tLoc  = hlsl::select(m0, t, hlsl::select(m1, t1, hlsl::select(m2, t2, t3)));

         xi = (wPick > density_type(0)) ? (tLoc / wPick) : domain_type(0);
         pdf *= wPick / wSum;

         const codomain_type childHeap   = codomain_type(4u) * W + codomain_type(1u) + codomain_type(slot);
         const bool          childIsLeaf = (w.childLeafMask & (1u << slot)) != 0u;
         if (childIsLeaf)
         {
            leaf_t leaf;
            leafAcc.template get<leaf_t, codomain_type>(childHeap - firstLeafIdx, leaf);
            cache.leaf = leaf;
            cache.pdf  = pdf;
            return childHeap;
         }
         W = childHeap;
      }
      return ~codomain_type(0);
   }

   density_type forwardPdf(const domain_type u_in, NBL_CONST_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC { return cache.pdf; }

   weight_type forwardWeight(const domain_type u_in, NBL_CONST_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC { return cache.pdf; }

   // Reconstruct the probability that generate() would have picked the leaf at heap index
   // `leafHeapIdx`. Walks the SAME known root->leaf path generate() took (no random pick), multiplying
   // the per-level child-selection ratio wSelf/wSum at each ancestor.
   //
   // Direction matters for cost, not correctness. generate() descends root->leaf and, at the TOPMOST
   // node where the MAX_RATIO "no clear winner" test fires, hands the leaf pick to the subtree alias --
   // so forward pdf = (descent ratios above the stop) * (subtree-alias pdf at the stop). We reproduce
   // that by descending the same direction and STOPPING at the first firing node: exactly one subtree-
   // alias lookup, and no node loads below the stop. (The older leaf->root climb always reached the
   // root and did an alias lookup at every firing level, discarding all but the topmost -- pure waste
   // on a latency-bound path, since "no clear winner" tends to fire high in the tree.) Only MAX_RATIO
   // is mirrored: PDF_FLOOR is non-local (depends on the from-root cumulative pdf), so forward/backward
   // agree only when PDF_FLOOR is disabled.
   //
   // The tree links parent<-child, so the root->leaf path isn't directly walkable. We pack it in one
   // bottom-up pass: each step's child slot is (h-1)&3, two bits, and a 4-ary heap over a 32-bit index
   // is at most 16 deep (4^16 ~= 2^32), so all 16 slots fit in a single uint32 -- no path array, no
   // O(depth^2) re-climb. Climbing leaf->root and shifting left each step lands the root's slot (climbed
   // last) in the LOW bit pair and the leaf's in the high pair, so the descent reads LSB-first and
   // rebuilds the node as 4*node+1+slot, exactly mirroring generate().
   density_type backwardPdf(const codomain_type leafHeapIdx) NBL_CONST_MEMBER_FUNC
   {
      if (leafHeapIdx == codomain_type(0))
         return density_type(1);
      if (firstLeafIdx == codomain_type(0))
         return density_type(0);

#if NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED
      // The scored leaf's array index, what the subtree alias indexes by.
      const codomain_type leafArrayIdx = leafHeapIdx - firstLeafIdx;
#endif

      // Pack the leaf->root slot sequence; shifting left each step puts the root's slot in the low pair.
      // parent(h) = (h-1)/4 < h for h >= 1, so this always reaches the root.
      uint32_t pathBits = 0u;
      uint32_t depth    = 0u;
      NBL_HLSL_LOOP
      for (codomain_type t = leafHeapIdx; t != codomain_type(0); t = (t - codomain_type(1)) / codomain_type(4))
      {
         pathBits = (pathBits << 2u) | uint32_t((t - codomain_type(1)) & codomain_type(3));
         ++depth;
      }

      density_type  pdf  = density_type(1);
      codomain_type node = codomain_type(0); // root
      NBL_HLSL_LOOP
      for (uint32_t level = 0u; level < depth; ++level)
      {
         // Root's slot is the low pair (packed last); walk up the pairs as we descend.
         const uint32_t slot = (pathBits >> (2u * level)) & 3u;

         wide_node_t w;
         nodeAcc.template get<wide_node_t, codomain_type>(node, w);

         // Stream the four child weights: backward only needs the running sum, the followed child's
         // weight, and (for MAX_RATIO) the running max -- never all four live at once the way
         // generate()'s CDF pick does, so this keeps fewer values resident.
         density_type wSum  = density_type(0);
         density_type wSelf = density_type(0);
#if NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED
         density_type wMax = density_type(0);
#endif
         for (uint32_t s = 0u; s < 4u; ++s)
         {
            const density_type ws = LightcutTreeChildWeight<T, Mode>::compute(w.children[s], shadingPoint, shadingNormal);
            wSum += ws;
            if (s == slot)
               wSelf = ws;
#if NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED
            wMax = hlsl::max(wMax, ws);
#endif
         }
         if (!(wSum > density_type(0)))
            return density_type(0);

#if NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED
         // Same "no clear winner" test generate() applies. The first (topmost) firing node is where
         // generate() handed off to the subtree alias, so multiply that alias pdf and stop -- the
         // ratios already accumulated above are exactly generate()'s descent above the stop.
         if (wMax < density_type(NBL_LIGHTCUT_TREE_STOP_MAX_RATIO) * wSum)
         {
            const density_type aliasPdf = subtreeAcc.backwardPdf(node, leafArrayIdx);
            if (!(aliasPdf > density_type(0)))
               return density_type(0);
            return pdf * aliasPdf;
         }
#endif

         pdf *= wSelf / wSum;
         node = codomain_type(4u) * node + codomain_type(1u) + codomain_type(slot); // descend to the followed child
      }
      return pdf;
   }

   weight_type backwardWeight(const codomain_type leafHeapIdx) NBL_CONST_MEMBER_FUNC { return backwardPdf(leafHeapIdx); }

   NodeAccessor         nodeAcc;
   LeafAccessor         leafAcc;
   SubtreeAliasAccessor subtreeAcc;
   codomain_type        firstLeafIdx;
   point_type           shadingPoint;
   point_type           shadingNormal;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
