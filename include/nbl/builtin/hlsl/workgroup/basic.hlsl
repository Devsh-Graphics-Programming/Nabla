// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BASIC_INCLUDED_

//! all functions must be called in uniform control flow (all workgroup invocations active)
namespace nbl
{
namespace hlsl
{
namespace workgroup
{
    static const uint MaxWorkgroupSizeLog2 = 11;
    static const uint MaxWorkgroupSize = 0x1u << MaxWorkgroupSizeLog2;
    
    bool Elect()
    {
        return gl_LocalInvocationIndex==0u;
    }
}
}
}
#endif
