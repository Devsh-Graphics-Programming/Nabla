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
    
static const uint32_t MinSubgroupSizeLog2 = 2u;
static const uint32_t MinSubgroupSize = 0x1u<<MinSubgroupSizeLog2;
static const uint32_t MaxSubgroupSizeLog2 = 6u;
static const uint32_t MaxSubgroupSize = 0x1u<<MaxSubgroupSizeLog2;

}
}
}

#endif