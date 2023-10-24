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

uint32_t gl_SubgroupSize() {
    return WaveGetLaneCount();
}

uint32_t gl_SubgroupSizeLog2() {
    return firstbithigh(gl_SubgroupSize());
}

uint32_t gl_SubgroupInvocationID() {
    return WaveGetLaneIndex();
}

uint32_t gl_SubgroupID() {
    // TODO (PentaKon): This is not always correct (subgroup IDs aren't always aligned with invocation index per the spec)
    return gl_LocalInvocationIndex >> gl_SubgroupSizeLog2();
}

uint32_t4 gl_SubgroupEqMask() {
    return uint32_t4(0,0,0,1) << gl_SubgroupInvocationID();
}

uint32_t4 gl_SubgroupGeMask() {
    return uint32_t4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu) << gl_SubgroupInvocationID();
}

uint32_t4 gl_SubgroupGtMask() {
    return gl_SubgroupGeMask() << 1;
}

uint32_t4 gl_SubgroupLeMask() {
    return ~gl_SubgroupGtMask();
}

uint32_t4 gl_SubgroupLtMask() {
    return ~gl_SubgroupGeMask();
}

bool subgroupElect() {
    return spirv::subgroupElect(/*subgroup execution scope*/ 3);
}

// Memory Semantics: AcquireRelease, UniformMemory, WorkgroupMemory, AtomicCounterMemory, ImageMemory
void subgroupBarrier() {
    // REVIEW-519: barrier with subgroup scope is not supported  so leave commented out for now 
    //spirv::controlBarrier(3, 3, 0x800 | 0x400 | 0x100 | 0x40 | 0x8);
}

void subgroupMemoryBarrierShared() {
    spirv::memoryBarrier(3, 0x800 | 0x400 | 0x100 | 0x40 | 0x8);
}

}
}
}

#endif