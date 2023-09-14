// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SHUFFLE_INCLUDED_

namespace nbl 
{
namespace hlsl
{
namespace spirv
{
template<typename T>
[[vk::ext_instruction(/* OpGroupNonUniformShuffle */ 345)]]
T groupShuffle(uint executionScope, T value, uint invocationId);

#ifdef NBL_GL_KHR_shader_subgroup_shuffle_relative
template<typename T>
[[vk::ext_capability(/* GroupNonUniformShuffleRelative */ 66)]]
[[vk::ext_instruction(/* OpGroupNonUniformShuffleUp */ 347)]]
T groupShuffleUp(uint executionScope, T value, uint delta);

template<typename T>
[[vk::ext_capability(/* GroupNonUniformShuffleRelative */ 66)]]
[[vk::ext_instruction(/* OpGroupNonUniformShuffleDown */ 348)]]
T groupShuffleDown(uint executionScope, T value, uint delta);
#endif
}
}
}

#endif