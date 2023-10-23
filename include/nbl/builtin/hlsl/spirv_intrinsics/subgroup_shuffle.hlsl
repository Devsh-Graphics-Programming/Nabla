// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_SHUFFLE_INCLUDED_


#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"


namespace nbl 
{
namespace hlsl
{
namespace spirv
{
template<typename T>
[[vk::ext_instruction( spv::OpGroupNonUniformShuffle  /*345*/)]]
T groupShuffle(uint executionScope, T value, uint invocationId);

#ifdef NBL_GL_KHR_shader_subgroup_shuffle_relative
template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformShuffleRelative  /*66*/)]]
[[vk::ext_instruction( spv::OpGroupNonUniformShuffleUp  /*347*/)]]
T groupShuffleUp(uint executionScope, T value, uint delta);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformShuffleRelative  /*66*/)]]
[[vk::ext_instruction( spv::OpGroupNonUniformShuffleDown  /*348*/)]]
T groupShuffleDown(uint executionScope, T value, uint delta);
#endif
}
}
}

#endif
