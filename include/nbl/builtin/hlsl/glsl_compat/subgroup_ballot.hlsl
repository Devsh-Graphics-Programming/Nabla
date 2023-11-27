// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/subgroup_ballot.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{

uint32_t4 gl_SubgroupEqMask()
{
    const uint32_t comp = gl_SubgroupInvocationID()>>5;
    uint32_t4 retval = uint32_t4(0,0,0,0);
    retval[comp] = 0x1u<<(gl_SubgroupInvocationID()&31u);
    return retval;
}

uint32_t4 gl_SubgroupGeMask()
{
    const uint32_t FullBits = 0xffffffffu;
    const uint32_t comp = gl_SubgroupInvocationID()>>5;
    uint32_t4 retval = uint32_t4(comp>0 ? 0u:FullBits,comp>1 ? 0u:FullBits,comp>2 ? 0u:FullBits,0u);
    retval[comp] = FullBits<<(gl_SubgroupInvocationID()&31u);
    return retval;
}

uint32_t4 gl_SubgroupGtMask()
{
    uint32_t4 retval = gl_SubgroupGeMask();
    const uint32_t comp = gl_SubgroupInvocationID()>>5;
    retval[comp] = 0xfffffffeu<<(gl_SubgroupInvocationID()&31u);
    return retval;
}

uint32_t4 gl_SubgroupLeMask() {
    return ~gl_SubgroupGtMask();
}

uint32_t4 gl_SubgroupLtMask() {
    return ~gl_SubgroupGeMask();
}

template<typename T>
T subgroupBroadcastFirst(T value)
{
    return spirv::subgroupBroadcastFirst<T>(spv::ScopeSubgroup, value);
}

template<typename T>
T subgroupBroadcast(T value, const uint32_t invocationId)
{
    return spirv::subgroupBroadcast<T>(spv::ScopeSubgroup, value, invocationId);
}

uint32_t4 subgroupBallot(bool value)
{
    return spirv::subgroupBallot(spv::ScopeSubgroup, value);
}

bool subgroupInverseBallot(uint32_t4 value)
{
    return spirv::subgroupInverseBallot(spv::ScopeSubgroup, value);
}

bool subgroupBallotBitExtract(uint32_t4 value, uint32_t index)
{
    return spirv::subgroupBallotBitExtract(spv::ScopeSubgroup, value, index);
}

uint32_t subgroupBallotBitCount(uint32_t4 value)
{
    return spirv::subgroupBallotBitCount(spv::ScopeSubgroup, spv::GroupOperationReduce, value);
}

uint32_t subgroupBallotInclusiveBitCount(uint32_t4 value)
{
    return spirv::subgroupBallotBitCount(spv::ScopeSubgroup, spv::GroupOperationInclusiveScan, value);
}

uint32_t subgroupBallotExclusiveBitCount(uint32_t4 value)
{
    return spirv::subgroupBallotBitCount(spv::ScopeSubgroup, spv::GroupOperationExclusiveScan, value);
}

uint32_t subgroupBallotFindLSB(uint32_t4 value)
{
    return spirv::subgroupBallotFindLSB(spv::ScopeSubgroup, value);
}

uint32_t subgroupBallotFindMSB(uint32_t4 value)
{
    return spirv::subgroupBallotFindMSB(spv::ScopeSubgroup, value);
}
}
}
}

#endif