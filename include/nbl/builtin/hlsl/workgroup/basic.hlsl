// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_

// TODO: `glsl::subgroupBroadcastFirst` is only used for its default `spv::ScopeSubgroup`, drop this dependency on the `glsl` namespace once the subgroup utilities move to a better namespace
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
    const uint16_t retval = (uint16_t(spirv::SubgroupId)<<glsl::gl_SubgroupSizeLog2())+uint16_t(spirv::SubgroupLocalInvocationId);
    assert(retval<Volume());
    return retval;
}
    
bool Elect()
{
    return spirv::SubgroupId==0 && spirv::SubgroupLocalInvocationId==0;
}

uint16_t ElectedSubgroupContiguousIndex()
{
    return glsl::subgroupBroadcastFirst<uint16_t>(SubgroupContiguousIndex());
}

}
}
}
#endif
