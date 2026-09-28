// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_LIGHTCUT_TREE_NODE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_LIGHTCUT_TREE_NODE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

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

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
