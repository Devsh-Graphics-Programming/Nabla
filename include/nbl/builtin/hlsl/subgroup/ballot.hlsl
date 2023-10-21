// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_ballot.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{

uint ElectedSubgroupInvocationID() {
    return glsl::subgroupBroadcastFirst<uint>(glsl::gl_SubgroupInvocationID());
}

uint ElectedLocalInvocationID() {
    return glsl::subgroupBroadcastFirst<uint>(gl_LocalInvocationIndex);
}

}
}
}

#endif