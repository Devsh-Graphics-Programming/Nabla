// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_ballot.hlsl"
#include "nbl/builtin/hlsl/subgroup/basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{

uint32_t LastSubgroupInvocation()
{
    // why this code was wrong before:
    // - only compute can use SubgroupID
    // - but there's no mapping of InvocationID to SubgroupID and Index
    return glsl::subgroupBallotFindMSB(subgroupBallot(true));
}

bool ElectLast()
{
    return glsl::gl_SubgroupInvocationID()==LastSubgroupInvocation();
}

template<typename T>
T BroadcastLast(T value)
{
    return glsl::subgroupBroadcast<T>(value,LastSubgroupInvocation());
}

uint32_t ElectedSubgroupInvocationID() {
    return glsl::subgroupBroadcastFirst<uint32_t>(glsl::gl_SubgroupInvocationID());
}

}
}
}

#endif