// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_BALLOT_INCLUDED_

namespace nbl 
{
namespace hlsl
{
namespace spirv
{
template<typename T>
[[vk::ext_capability(/* GroupNonUniformBallot */ 64)]]
[[vk::ext_instruction(/* OpGroupNonUniformBroadcastFirst */ 338)]]
T subgroupBroadcastFirst(uint executionScope, T value);

template<typename T>
[[vk::ext_capability(/* GroupNonUniformBallot */ 64)]]
[[vk::ext_instruction(/* OpGroupNonUniformBroadcast */ 337)]]
T subgroupBroadcast(uint executionScope, T value, uint invocationId);
}
}
}

#endif