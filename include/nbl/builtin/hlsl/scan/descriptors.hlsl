// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_SCAN_DESCRIPTORS_INCLUDED_
#define _NBL_HLSL_SCAN_DESCRIPTORS_INCLUDED_

#include "nbl/builtin/hlsl/scan/declarations.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

// coherent -> globallycoherent

namespace nbl
{
namespace hlsl
{
namespace scan
{

template<uint32_t dataElementCount=SCRATCH_EL_CNT - NBL_BUILTIN_MAX_LEVELS>
struct Scratch
{
    uint32_t reduceResult;
    uint32_t workgroupsStarted[NBL_BUILTIN_MAX_LEVELS];
    uint32_t data[dataElementCount];
};

[[vk::binding(0 ,0)]] RWStructuredBuffer<Storage_t> scanBuffer; // (REVIEW): Make the type externalizable. Decide how (#define?)
[[vk::binding(1 ,0)]] RWStructuredBuffer<Scratch> /*globallycoherent (seems we can't use along with VMM)*/ scanScratchBuf; // (REVIEW): Check if globallycoherent can be used with Vulkan Mem Model

template<typename Storage_t, bool isExclusive>
void getData(
    NBL_REF_ARG(Storage_t) data,
    NBL_CONST_REF_ARG(uint32_t) levelInvocationIndex,
    NBL_CONST_REF_ARG(uint32_t) levelWorkgroupIndex,
    NBL_CONST_REF_ARG(uint32_t) treeLevel
)
{
    const Parameters_t params = getParameters(); // defined differently for direct and indirect shaders
    
    uint32_t offset = levelInvocationIndex;
    const bool notFirstOrLastLevel = bool(treeLevel);
    if (notFirstOrLastLevel)
        offset += params.temporaryStorageOffset[treeLevel-1u];
    
    //if (pseudoLevel!=treeLevel) // downsweep/scan
    //{
    //    const bool firstInvocationInGroup = workgroup::SubgroupContiguousIndex()==0u;
    //    if (bool(levelWorkgroupIndex) && firstInvocationInGroup)
    //        data = scanScratchBuf[0].data[levelWorkgroupIndex+params.temporaryStorageOffset[treeLevel]];
    //
    //    if (notFirstOrLastLevel)
    //    {
    //        if (!firstInvocationInGroup)
    //            data = scanScratchBuf[0].data[offset-1u];
    //    }
    //    else
    //    {
    //        if(isExclusive)
    //        {
    //            if (!firstInvocationInGroup)
    //                data += scanBuffer[offset-1u];
    //        }
    //        else
    //        {
    //            data += scanBuffer[offset];
    //        }
    //    }
    //}
    //else
    //{
        if (notFirstOrLastLevel)
            data = scanScratchBuf[0].data[offset];
        else
            data = scanBuffer[offset];
    //}
}

template<typename Storage_t, bool isScan>
void setData(
    NBL_CONST_REF_ARG(Storage_t) data,
    NBL_CONST_REF_ARG(uint32_t) levelInvocationIndex,
    NBL_CONST_REF_ARG(uint32_t) levelWorkgroupIndex,
    NBL_CONST_REF_ARG(uint32_t) treeLevel,
    NBL_CONST_REF_ARG(bool) inRange
)
{
    const Parameters_t params = getParameters();
    if (!isScan && treeLevel<params.topLevel) // is reduce and we're not at the last level (i.e. we still save into scratch)
    {
        const bool lastInvocationInGroup = workgroup::SubgroupContiguousIndex()==(glsl::gl_WorkGroupSize().x-1u);
        if (lastInvocationInGroup)
            scanScratchBuf[0u].data[levelWorkgroupIndex+params.temporaryStorageOffset[treeLevel]] = data;
    }
    else if (inRange)
    {
        if (!isScan && treeLevel == params.topLevel)
        {
            scanScratchBuf[0u].reduceResult = data;
        }
        // The following only for isScan == true
        else if (bool(treeLevel))
        {
            const uint32_t offset = params.temporaryStorageOffset[treeLevel-1u];
            scanScratchBuf[0].data[levelInvocationIndex+offset] = data;
        }
        else
        {
            scanBuffer[levelInvocationIndex] = data;
        }
    }
}

}
}
}

#endif