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
namespace impl
{
[[vk::ext_capability(/* GroupNonUniformBallot */ 64)]]
void spirv_ballot_cap(){}
}
}
}
}

#endif