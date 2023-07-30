// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"

// [[vk::ext_extension("GL_KHR_shader_subgroup_ballot")]] REVIEW-519: Extensions don't seem to be needed?
[[vk::ext_capability(/* GroupNonUniformBallot */ 64)]]
void spirv_ballot_cap(){}

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{

uint ElectedSubgroupInvocationID() {
	return glsl::subgroupBroadcastFirst<uint>(InvocationID());
}

uint ElectedLocalInvocationID() {
	return glsl::subgroupBroadcastFirst<uint>(gl_LocalInvocationIndex);
}

}
}
}

#endif