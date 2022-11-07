// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_UTILS_INDIRECT_COMMANDS_GLSL_INCLUDED_
#define _NBL_BUILTIN_UTILS_INDIRECT_COMMANDS_GLSL_INCLUDED_


struct nbl_glsl_DrawArraysIndirectCommand_t
{
    uint  count;
    uint  instanceCount;
    uint  first;
    uint  baseInstance;
};

struct nbl_glsl_DrawElementsIndirectCommand_t
{
    uint count;
    uint instanceCount;
    uint firstIndex;
    uint baseVertex;
    uint baseInstance;
};

struct nbl_glsl_DispatchIndirectCommand_t
{
    uint  num_groups_x;
    uint  num_groups_y;
    uint  num_groups_z;
};


uint nbl_glsl_utils_computeOptimalPersistentWorkgroupDispatchSize(in uint elementCount, in uint workgroupSize, in uint workgroupSpinningProtection)
{
    const uint infinitelyWideDeviceWGCount = (elementCount-1u)/(workgroupSize*workgroupSpinningProtection)+1u;
    return min(infinitelyWideDeviceWGCount,NBL_GLSL_LIMIT_MAX_RESIDENT_INVOCATIONS/NBL_GLSL_LIMIT_MAX_OPTIMALLY_RESIDENT_WORKGROUP_INVOCATIONS);
}
uint nbl_glsl_utils_computeOptimalPersistentWorkgroupDispatchSize(in uint elementCount, in uint workgroupSize)
{
    return nbl_glsl_utils_computeOptimalPersistentWorkgroupDispatchSize(elementCount,workgroupSize,1u);
}


#endif