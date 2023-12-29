// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_SHUFFLE_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/subgroup_shuffle.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{
template<typename T>
T subgroupShuffle(T value, uint32_t invocationId)
{
    return spirv::groupShuffle<T>(spv::ScopeSubgroup, value, invocationId);
}

template<typename T>
T subgroupShuffleUp(T value, uint32_t delta)
{
    return spirv::groupShuffleUp<T>(spv::ScopeSubgroup, value, delta);
}

template<typename T>
T subgroupShuffleDown(T value, uint32_t delta)
{
    return spirv::groupShuffleDown<T>(spv::ScopeSubgroup, value, delta);
}

}
}
}

#endif