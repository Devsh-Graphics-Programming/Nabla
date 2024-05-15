// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_BALLOT_INCLUDED_


#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/subgroup_basic.hlsl"


namespace nbl 
{
namespace hlsl
{
namespace spirv
{
[[vk::ext_builtin_input(spv::BuiltInSubgroupEqMask)]]
static const uint32_t4 BuiltInSubgroupEqMask;
[[vk::ext_builtin_input(spv::BuiltInSubgroupGeMask)]]
static const uint32_t4 BuiltInSubgroupGeMask;
[[vk::ext_builtin_input(spv::BuiltInSubgroupGtMask)]]
static const uint32_t4 BuiltInSubgroupGtMask;
[[vk::ext_builtin_input(spv::BuiltInSubgroupLeMask)]]
static const uint32_t4 BuiltInSubgroupLeMask;
[[vk::ext_builtin_input(spv::BuiltInSubgroupLtMask)]]
static const uint32_t4 BuiltInSubgroupLtMask;

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformBallot )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBroadcastFirst )]]
T subgroupBroadcastFirst(uint32_t executionScope, T value);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformBallot )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBroadcast )]]
T subgroupBroadcast(uint32_t executionScope, T value, uint32_t invocationId);

[[vk::ext_capability( spv::CapabilityGroupNonUniformBallot )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBallot )]]
uint32_t4 subgroupBallot(uint32_t executionScope, bool value);

[[vk::ext_capability( spv::CapabilityGroupNonUniformBallot )]]
[[vk::ext_instruction( spv::OpGroupNonUniformInverseBallot )]]
bool subgroupInverseBallot(uint32_t executionScope, uint32_t4 value);

[[vk::ext_capability( spv::CapabilityGroupNonUniformBallot )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBallotBitExtract )]]
bool subgroupBallotBitExtract(uint32_t executionScope, uint32_t4 value, uint32_t id);

[[vk::ext_capability( spv::CapabilityGroupNonUniformBallot )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBallotBitCount )]]
uint32_t subgroupBallotBitCount(uint32_t executionScope, [[vk::ext_literal]] uint32_t operation, uint32_t4 value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBallotFindLSB)]]
uint32_t subgroupBallotFindLSB(uint32_t executionScope, uint32_t4 value);

[[vk::ext_capability(spv::CapabilityGroupNonUniformBallot)]]
[[vk::ext_instruction(spv::OpGroupNonUniformBallotFindMSB)]]
uint32_t subgroupBallotFindMSB(uint32_t executionScope, uint32_t4 value);
}
}
}

#endif
