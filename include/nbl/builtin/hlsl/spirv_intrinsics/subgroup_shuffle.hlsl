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
[[vk::ext_instruction( spv::OpGroupNonUniformShuffle )]]
T groupShuffle(uint32_t executionScope, T value, uint32_t invocationId);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformShuffleRelative )]]
[[vk::ext_instruction( spv::OpGroupNonUniformShuffleUp )]]
T groupShuffleUp(uint32_t executionScope, T value, uint32_t delta);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformShuffleRelative )]]
[[vk::ext_instruction( spv::OpGroupNonUniformShuffleDown )]]
T groupShuffleDown(uint32_t executionScope, T value, uint32_t delta);

}
}
}

#endif
