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
T subgroupShuffle(T value, uint invocationId)
{
    return spirv::groupShuffle<T>(3, value, invocationId);
}

template<typename T>
T subgroupShuffleUp(T value, uint delta)
{
#ifdef NBL_GL_KHR_shader_subgroup_shuffle_relative
    return spirv::groupShuffleUp<T>(3, value, delta);
#else
    return spirv::groupShuffle<T>(3, value, gl_SubgroupInvocationID() - delta);
#endif
}

template<typename T>
T subgroupShuffleDown(T value, uint delta)
{
#ifdef NBL_GL_KHR_shader_subgroup_shuffle_relative
    return spirv::groupShuffleDown<T>(3, value, delta);
#else
    return spirv::groupShuffle<T>(3, value, gl_SubgroupInvocationID() + delta);
#endif
}

}
}
}

#endif