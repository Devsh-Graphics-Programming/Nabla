// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_BALLOT_INCLUDED_


#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"


namespace nbl 
{
namespace hlsl
{
namespace spirv
{
template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformBallot )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBroadcastFirst )]]
T subgroupBroadcastFirst(uint32_t executionScope, T value);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformBallot )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBroadcast )]]
T subgroupBroadcast(uint32_t executionScope, T value, uint32_t invocationId);
}
}
}

#endif
