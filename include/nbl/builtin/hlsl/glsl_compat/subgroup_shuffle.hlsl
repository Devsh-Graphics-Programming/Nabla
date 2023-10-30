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
<<<<<<< HEAD
T subgroupShuffle(T value, uint invocationId)
=======
T subgroupShuffle(T value, uint32_t invocationId)
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
{
    return spirv::groupShuffle<T>(spv::ScopeSubgroup, value, invocationId);
}

template<typename T>
<<<<<<< HEAD
T subgroupShuffleUp(T value, uint delta)
=======
T subgroupShuffleUp(T value, uint32_t delta)
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
{
#ifdef NBL_GL_KHR_shader_subgroup_shuffle_relative
    return spirv::groupShuffleUp<T>(spv::ScopeSubgroup, value, delta);
#else
    return spirv::groupShuffle<T>(spv::ScopeSubgroup, value, gl_SubgroupInvocationID() - delta);
#endif
}

template<typename T>
<<<<<<< HEAD
T subgroupShuffleDown(T value, uint delta)
=======
T subgroupShuffleDown(T value, uint32_t delta)
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
{
#ifdef NBL_GL_KHR_shader_subgroup_shuffle_relative
    return spirv::groupShuffleDown<T>(spv::ScopeSubgroup, value, delta);
#else
    return spirv::groupShuffle<T>(spv::ScopeSubgroup, value, gl_SubgroupInvocationID() + delta);
#endif
}

}
}
}

#endif