// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_BASIC_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/subgroup_basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{

#ifdef __HLSL_VERSION
uint32_t gl_SubgroupSize() {
    return WaveGetLaneCount();
}

uint32_t gl_SubgroupSizeLog2() {
    return firstbithigh(gl_SubgroupSize());
}

uint32_t gl_SubgroupInvocationID() {
    return WaveGetLaneIndex();
}

// only available in compute
uint32_t gl_SubgroupID() {
    // TODO (PentaKon): This is not always correct (subgroup IDs aren't always aligned with invocation index per the spec)
    return gl_LocalInvocationIndex() >> gl_SubgroupSizeLog2();
}

bool subgroupElect() {
    return spirv::subgroupElect(spv::ScopeSubgroup);
}

void subgroupBarrier() {
    spirv::controlBarrier(spv::ScopeSubgroup, spv::ScopeSubgroup, spv::MemorySemanticsImageMemoryMask | spv::MemorySemanticsWorkgroupMemoryMask | spv::MemorySemanticsUniformMemoryMask | spv::MemorySemanticsAcquireReleaseMask);
}

void subgroupMemoryBarrier() {
    spirv::memoryBarrier(spv::ScopeSubgroup, spv::MemorySemanticsImageMemoryMask | spv::MemorySemanticsWorkgroupMemoryMask | spv::MemorySemanticsUniformMemoryMask | spv::MemorySemanticsAcquireReleaseMask);
}

void subgroupMemoryBarrierBuffer() {
    spirv::memoryBarrier(spv::ScopeSubgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
}

void subgroupMemoryBarrierShared() {
    spirv::memoryBarrier(spv::ScopeSubgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask);
}

void subgroupMemoryBarrierImage() {
    spirv::memoryBarrier(spv::ScopeSubgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsImageMemoryMask);
}
#endif

}
}
}

#endif