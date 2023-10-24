// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BASIC_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{
    
static const uint32_t MinSubgroupSizeLog2 = 4;
static const uint32_t MinSubgroupSize = 0x1u << MinSubgroupSizeLog2;
    
uint32_t LastSubgroupInvocation() {
    uint32_t lastSubgroupInvocation = glsl::gl_SubgroupSize() - 1u;
    if(glsl::gl_SubgroupID() == ((_NBL_HLSL_WORKGROUP_SIZE_ - 1u) >> glsl::gl_SubgroupSizeLog2())) {
        lastSubgroupInvocation &= _NBL_HLSL_WORKGROUP_SIZE_ - 1u; // if workgroup size is not a multiple of subgroup then we return the remainder of the division of the last workgroup invocation index by subgroup size which is then the index of the last active invocation in the last subgroup
    }
    return lastSubgroupInvocation;
}

}
}
}

#endif