// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_BASIC_INCLUDED_


#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"


namespace nbl 
{
namespace hlsl
{
namespace spirv
{
[[vk::ext_builtin_input(spv::BuiltInSubgroupSize)]]
static const uint32_t SubgroupSize;
[[vk::ext_builtin_input(spv::BuiltInNumSubgroups)]]
static const uint32_t NumSubgroups;
[[vk::ext_builtin_input(spv::BuiltInSubgroupId)]]
static const uint32_t SubgroupId;
[[vk::ext_builtin_input(spv::BuiltInSubgroupLocalInvocationId)]]
static const uint32_t SubgroupLocalInvocationId;

[[vk::ext_instruction( spv::OpGroupNonUniformElect )]]
bool subgroupElect(uint32_t executionScope);
}
}
}

#endif
