// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_SCAN_DECLARATIONS_INCLUDED_
#define _NBL_HLSL_SCAN_DECLARATIONS_INCLUDED_

// REVIEW: Not sure if this file is needed in HLSL implementation

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#ifndef NBL_BUILTIN_MAX_LEVELS
#define NBL_BUILTIN_MAX_LEVELS 7
#endif

namespace nbl
{
namespace hlsl
{
namespace scan
{
    // REVIEW: Putting topLevel second allows better alignment for packing of constant variables, assuming lastElement has length 4. (https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules)
    struct Parameters_t {
        uint32_t lastElement[NBL_BUILTIN_MAX_LEVELS/2+1];
        uint32_t topLevel;
        uint32_t temporaryStorageOffset[NBL_BUILTIN_MAX_LEVELS/2];
    };
    
    Parameters_t getParameters();

    struct DefaultSchedulerParameters_t
    {
        uint32_t cumulativeWorkgroupCount[NBL_BUILTIN_MAX_LEVELS];
        uint32_t workgroupFinishFlagsOffset[NBL_BUILTIN_MAX_LEVELS];
        uint32_t lastWorkgroupSetCountForLevel[NBL_BUILTIN_MAX_LEVELS];

    };
    
    DefaultSchedulerParameters_t getSchedulerParameters();

    template<typename Storage_t, bool isExclusive=false>
    void getData(
        NBL_REF_ARG(Storage_t) data,
        NBL_CONST_REF_ARG(uint32_t) levelInvocationIndex,
        NBL_CONST_REF_ARG(uint32_t) localWorkgroupIndex,
        NBL_CONST_REF_ARG(uint32_t) treeLevel,
        NBL_CONST_REF_ARG(uint32_t) pseudoLevel
    );

    template<typename Storage_t, bool isScan>
    void setData(
        NBL_CONST_REF_ARG(Storage_t) data,
        NBL_CONST_REF_ARG(uint32_t) levelInvocationIndex,
        NBL_CONST_REF_ARG(uint32_t) localWorkgroupIndex,
        NBL_CONST_REF_ARG(uint32_t) treeLevel,
        NBL_CONST_REF_ARG(uint32_t) pseudoLevel,
        NBL_CONST_REF_ARG(bool) inRange
    );
    
}
}
}

#endif