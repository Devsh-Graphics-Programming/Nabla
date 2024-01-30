// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_SCAN_DESCRIPTORS_INCLUDED_
#define _NBL_HLSL_SCAN_DESCRIPTORS_INCLUDED_

#include "nbl/builtin/hlsl/scan/declarations.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

// coherent -> globallycoherent

// (REVIEW): This should be externally defined. Maybe change the scratch buffer to RWByteAddressBuffer? Annoying to manage though...
#ifndef SCRATCH_SIZE
#error "Must manually define SCRATCH_SIZE for now"
#endif

namespace nbl
{
namespace hlsl
{
namespace scan
{

template<uint32_t scratchElementCount=SCRATCH_SIZE>
struct Scratch
{
    uint32_t workgroupsStarted;
    uint32_t data[scratchElementCount];
};

[[vk::binding(0 ,0)]] StructuredBuffer<uint32_t> scanInputBuf; // (REVIEW): Make the type externalizable. Decide how (#define?)
[[vk::binding(1 ,0)]] RWStructuredBuffer<Scratch> globallycoherent scanScratchBuf; // (REVIEW): Check if globallycoherent can be used with Vulkan Mem Model

template<typename Storage_t, bool isExclusive=false>
void getData(
    NBL_REF_ARG(Storage_t) data,
    NBL_CONST_REF_ARG(uint32_t) levelInvocationIndex,
    NBL_CONST_REF_ARG(uint32_t) localWorkgroupIndex,
    NBL_CONST_REF_ARG(uint32_t) treeLevel,
    NBL_CONST_REF_ARG(uint32_t) pseudoLevel
)
{
    const Parameters_t params = getParameters(); // defined differently for direct and indirect shaders
    
    uint32_t offset = levelInvocationIndex;
    const bool notFirstOrLastLevel = bool(pseudoLevel);
    if (notFirstOrLastLevel)
		offset += params.temporaryStorageOffset[pseudoLevel-1u];
    
    if (pseudoLevel!=treeLevel) // downsweep
	{
		const bool firstInvocationInGroup = SubgroupContiguousIndex()==0u;
		if (bool(localWorkgroupIndex) && firstInvocationInGroup)
			data = scanScratchBuf[0].data[localWorkgroupIndex+params.temporaryStorageOffset[pseudoLevel]];

		if (notFirstOrLastLevel)
		{
			if (!firstInvocationInGroup)
				data = scanScratchBuf[0].data[offset-1u];
		}
		else
		{
            if(isExclusive)
            {
                if (!firstInvocationInGroup)
                    data += scanInputBuf[offset-1u];
            }
            else
            {
                data += scanInputBuf[offset];
            }
		}
	}
	else
	{
		if (notFirstOrLastLevel)
			data = scanScratchBuf[0].data[offset];
		else
			data = scanInputBuf[offset];
	}
}

template<typename Storage_t>
void setData(
    NBL_CONST_REF_ARG(Storage_t) data,
    NBL_CONST_REF_ARG(uint32_t) levelInvocationIndex,
    NBL_CONST_REF_ARG(uint32_t) localWorkgroupIndex,
    NBL_CONST_REF_ARG(uint32_t) treeLevel,
    NBL_CONST_REF_ARG(uint32_t) pseudoLevel,
    NBL_CONST_REF_ARG(bool) inRange
)
{
    const Parameters_t params = getParameters();
	if (treeLevel<params.topLevel)
	{
		const bool lastInvocationInGroup = SubgroupContiguousIndex()==(glsl::gl_WorkGroupSize().x-1);
		if (lastInvocationInGroup)
			scanScratchBuf[0].data[localWorkgroupIndex+params.temporaryStorageOffset[treeLevel]] = data;
	}
	else if (inRange)
	{
		if (bool(pseudoLevel))
		{
			const uint32_t offset = params.temporaryStorageOffset[pseudoLevel-1u];
			scanScratchBuf[0].data[levelInvocationIndex+offset] = data;
		}
		else
			scanInputBuf[levelInvocationIndex] = data;
	}
}

}
}
}

#endif