
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_UTILS_INDIRECT_COMMANDS_HLSL_INCLUDED_
#define _NBL_BUILTIN_UTILS_INDIRECT_COMMANDS_HLSL_INCLUDED_

namespace nbl
{
namespace hlsl
{

struct DrawArraysIndirectCommand_t
{
    uint  count;
    uint  instanceCount;
    uint  first;
    uint  baseInstance;
};

struct DrawElementsIndirectCommand_t
{
    uint count;
    uint instanceCount;
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
};

struct DispatchIndirectCommand_t
{
    uint  num_groups_x;
    uint  num_groups_y;
    uint  num_groups_z;
};


uint computeOptimalPersistentWorkgroupDispatchSize(in uint elementCount, in uint workgroupSize, in uint workgroupSpinningProtection)
{
    const uint infinitelyWideDeviceWGCount = (elementCount-1u)/(workgroupSize*workgroupSpinningProtection)+1u;
    return min(infinitelyWideDeviceWGCount,NBL_HLSL_LIMIT_MAX_RESIDENT_INVOCATIONS/NBL_HLSL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS);
}
uint computeOptimalPersistentWorkgroupDispatchSize(in uint elementCount, in uint workgroupSize)
{
    return computeOptimalPersistentWorkgroupDispatchSize(elementCount,workgroupSize,1u);
}


}
}

#endif