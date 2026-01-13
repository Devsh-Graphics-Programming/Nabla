// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_VOTE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_VOTE_INCLUDED_


#include "nbl/builtin/hlsl/spirv_intrinsics/subgroup_basic.hlsl"


namespace nbl 
{
namespace hlsl
{
namespace spirv
{

[[vk::ext_instruction( spv::OpGroupNonUniformAll)]]
bool subgroupAll(uint32_t groupScope, bool value);

[[vk::ext_instruction(spv::OpGroupNonUniformAny)]]
bool subgroupAny(uint32_t groupScope, bool value);

template<typename T>
[[vk::ext_instruction(spv::OpGroupNonUniformAllEqual)]]
bool subgroupAllEqual(uint32_t groupScope, T value);

}
}
}

#endif
