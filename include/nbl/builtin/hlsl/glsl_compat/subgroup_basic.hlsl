// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_BASIC_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/subgroup_basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{

// TODO (Future): Accessing gl_SubgroupSize and other gl_* values is not yet possible due to https://github.com/microsoft/DirectXShaderCompiler/issues/4217

uint gl_SubgroupSize() {
    return WaveGetLaneCount();
}

uint gl_SubgroupSizeLog2() {
    return firstbithigh(gl_SubgroupSize());
}

uint gl_SubgroupInvocationID() {
    return WaveGetLaneIndex();
}

uint gl_SubgroupID() {
    // TODO (PentaKon): This is not always correct (subgroup IDs aren't always aligned with invocation index per the spec)
    return gl_LocalInvocationIndex >> gl_SubgroupSizeLog2();
}

uint4 gl_SubgroupEqMask() {
    return uint4(0,0,0,1) << gl_SubgroupInvocationID();
}

uint4 gl_SubgroupGeMask() {
    return uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu) << gl_SubgroupInvocationID();
}

uint4 gl_SubgroupGtMask() {
    return gl_SubgroupGeMask() << 1;
}

uint4 gl_SubgroupLeMask() {
    return ~gl_SubgroupGtMask();
}

uint4 gl_SubgroupLtMask() {
    return ~gl_SubgroupGeMask();
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

}
}
}

#endif