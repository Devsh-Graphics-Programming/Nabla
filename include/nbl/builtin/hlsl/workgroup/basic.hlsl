// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_

#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

//! all functions must be called in uniform control flow (all workgroup invocations active)
namespace nbl
{
namespace hlsl
{
namespace workgroup
{
    static const uint MaxWorkgroupSizeLog2 = 11;
    static const uint MaxWorkgroupSize = 0x1u << MaxWorkgroupSizeLog2;
    
    uint SubgroupContiguousIndex()
    {
        return glsl::gl_SubgroupID() * glsl::gl_SubgroupSize() + glsl::gl_SubgroupInvocationID();
    }
    
    bool Elect()
    {
        return SubgroupContiguousIndex()==0u;
    }
}
}
}
#endif
