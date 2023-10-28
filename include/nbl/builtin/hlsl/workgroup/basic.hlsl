// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_ballot.hlsl"

//! all functions must be called in uniform control flow (all workgroup invocations active)
namespace nbl
{
namespace hlsl
{
namespace workgroup
{

static const uint32_t MaxWorkgroupSizeLog2 = 11;
static const uint32_t MaxWorkgroupSize = 0x1u<<MaxWorkgroupSizeLog2;

uint16_t Volume()
{
    const uint16_t3 dims = uint16_t3(glsl::gl_WorkGroupSize());
    return dims.x*dims.y*dims.z;
}
    
uint16_t SubgroupContiguousIndex()
{
    return uint16_t(glsl::gl_SubgroupID()*glsl::gl_SubgroupSize()+glsl::gl_SubgroupInvocationID());
}
    
bool Elect()
{
    return glsl::gl_SubgroupID()==0 && glsl::gl_SubgroupInvocationID()==0;
}

uint16_t ElectedSubgroupContiguousIndex()
{
    return glsl::subgroupBroadcastFirst<uint16_t>(SubgroupContiguousIndex());
}

}
}
}
#endif
