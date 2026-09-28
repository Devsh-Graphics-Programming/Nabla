// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_LIGHTCUT_TREE_PACKING_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_LIGHTCUT_TREE_PACKING_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>
#include <nbl/builtin/hlsl/sampling/lightcut_tree/node.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

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

   // Must unroll: a rolled loop indexes `decoded.children` dynamically, forcing the
   // array into Function storage (scratch) instead of scalarizing into registers.
   NBL_UNROLL
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

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
