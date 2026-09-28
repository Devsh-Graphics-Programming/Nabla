// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_LIGHTCUT_TREE_SAMPLER_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_LIGHTCUT_TREE_SAMPLER_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/concepts/core.hlsl>
#include <nbl/builtin/hlsl/sampling/lightcut_tree/node.hlsl>
#include <nbl/builtin/hlsl/sampling/lightcut_tree/weight.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

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

   codomain_type generate(const domain_type u_in) NBL_CONST_MEMBER_FUNC
   {
      cache_type cache;
      return generate(u_in, cache);
   }

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
      // 16 is a hard bound: a 4-ary heap over a 32-bit index is at most 16 deep.
      // The fallback return handles malformed trees that never set a leaf bit.
      NBL_HLSL_LOOP
      for (uint32_t step = 0u; step < 16u; ++step)
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

         // Fallback must be the last child with w > 0 (wSum > 0 guarantees one): rounding can
         // push t past w0+w1+w2 with w3 == 0 (padding/culled), and that child need not exist.
         const bool         p1      = w1 > density_type(0);
         const bool         p2      = w2 > density_type(0);
         const bool         p3      = w3 > density_type(0);
         const uint32_t     lastPos = hlsl::select(p3, 3u, hlsl::select(p2, 2u, hlsl::select(p1, 1u, 0u)));
         const density_type wLast   = hlsl::select(p3, w3, hlsl::select(p2, w2, hlsl::select(p1, w1, w0)));
         const density_type tLast   = hlsl::select(p3, t3, hlsl::select(p2, t2, hlsl::select(p1, t1, t)));

         const uint32_t     slot  = hlsl::select(m0, 0u, hlsl::select(m1, 1u, hlsl::select(m2, 2u, lastPos)));
         const density_type wPick = hlsl::select(m0, w0, hlsl::select(m1, w1, hlsl::select(m2, w2, wLast)));
         const density_type tLoc  = hlsl::select(m0, t, hlsl::select(m1, t1, hlsl::select(m2, t2, tLast)));

         // clamp: tLoc overshoots wPick on that fallback.
         xi = (wPick > density_type(0)) ? hlsl::clamp(tLoc / wPick, domain_type(0), domain_type(1)) : domain_type(0);
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

         density_type wSum  = density_type(0);
         density_type wSelf = density_type(0);
#if NBL_LIGHTCUT_TREE_STOP_MAX_RATIO_ENABLED
         density_type wMax = density_type(0);
#endif
         // Must unroll, same reason as the decode in packing.hlsl.
         NBL_UNROLL
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
         // generate() handed off to the subtree alias, so multiply that alias pdf and stop, the
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
