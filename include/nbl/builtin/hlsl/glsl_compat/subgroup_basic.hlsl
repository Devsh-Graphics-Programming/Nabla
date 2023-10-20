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
    return spirv::subgroupElect(/*subgroup execution scope*/ 3);
}

void subgroupBarrier() {
    spirv::controlBarrier(3, 3, SpvMemorySemanticsImageMemoryMask | SpvMemorySemanticsWorkgroupMemoryMask | SpvMemorySemanticsUniformMemoryMask | SpvMemorySemanticsAcquireReleaseMask);
}

void subgroupMemoryBarrier() {
    spirv::memoryBarrier(3, SpvMemorySemanticsImageMemoryMask | SpvMemorySemanticsWorkgroupMemoryMask | SpvMemorySemanticsUniformMemoryMask | SpvMemorySemanticsAcquireReleaseMask);
}

void subgroupMemoryBarrierBuffer() {
    spirv::memoryBarrier(3, SpvMemorySemanticsAcquireReleaseMask | SpvMemorySemanticsUniformMemoryMask);
}

void subgroupMemoryBarrierShared() {
    spirv::memoryBarrier(3, SpvMemorySemanticsAcquireReleaseMask | SpvMemorySemanticsWorkgroupMemoryMask);
}

void subgroupMemoryBarrierImage() {
    spirv::memoryBarrier(3, SpvMemorySemanticsAcquireReleaseMask | SpvMemorySemanticsImageMemoryMask);
}

}
}
}

#endif