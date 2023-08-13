// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_INTRINSICS_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_INTRINSICS_BASIC_INCLUDED_

namespace nbl 
{
namespace hlsl
{
namespace spirv
{
[[vk::ext_instruction(/* OpGroupNonUniformElect */ 333)]]
bool subgroupElect([[vk::ext_literal]] uint executionScope);
}
}
}

#endif