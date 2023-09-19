// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat/cpp_compat.h"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

template<typename T, class SubgroupOp, class SharedAccessor>
struct WorkgroupScanHead
{
    T identity;
    T firstLevelScan;
    T lastLevelScan;
    uint itemCount;
    uint lastInvocation;
    uint lastInvocationInLevel;
    uint scanLoadIndex;

    static WorkgroupScanHead create(T identity, uint itemCount)
    {
        WorkgroupScanHead<T, SubgroupOp, SharedAccessor> wsh;
        wsh.identity = identity;
        wsh.itemCount = itemCount;
        wsh.lastInvocation = itemCount - 1u;
        return wsh;
    }

    void operator()(T value, NBL_REF_ARG(SharedAccessor) sharedAccessor)
    {
        const uint subgroupMask = glsl::gl_SubgroupSize() - 1u;
        lastInvocationInLevel = lastInvocation;
        
        SubgroupOp subgroupOp;
        firstLevelScan = subgroupOp(value);
        T scan = firstLevelScan;
        
        const bool isLastSubgroupInvocation = glsl::gl_SubgroupInvocationID() == glsl::gl_SubgroupSize() - 1u;
        
        // Since we are scanning the RESULT of the initial scan (which paired one input per subgroup invocation) 
        // every group of 64 invocations has been coallesced into 1 result value. This means that the results of 
        // the first SubgroupSz^2 invocations will be processed by the first subgroup and so on.
        // Consequently, those first SubgroupSz^2 invocations will store their results on SubgroupSz scratch slots 
        // and the next level will follow the same + the previous as an `offset`.
        
        scanLoadIndex = gl_LocalInvocationIndex;
        const uint loadStoreIndexDiff = scanLoadIndex - glsl::gl_SubgroupID();
        
        bool participate = gl_LocalInvocationIndex <= lastInvocationInLevel;
        while(lastInvocationInLevel >= glsl::gl_SubgroupSize() * glsl::gl_SubgroupSize())
        {
            // REVIEW-519: Refactor to use ping-ponging if possible in order to minimize barrier usage. Worth it? How common is WGSZ > 4096
            if(participate)
            {
                if (any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, isLastSubgroupInvocation)))
                { // only invocations that have the final value of the subgroupOp (inclusive scan) store their results
                    sharedAccessor.main.set(scanLoadIndex - loadStoreIndexDiff, scan); // For subgroupSz = 64, first 4095 invocations store index is [0,63], 4096-8191 [64,127] etc.
                }
            }
            sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
            participate = gl_LocalInvocationIndex <= (lastInvocationInLevel >>= glsl::gl_SubgroupSizeLog2());
            if(participate)
            {
                const uint prevLevelScan = sharedAccessor.main.get(gl_LocalInvocationIndex);
                scan = subgroupOp(prevLevelScan);
                sharedAccessor.main.set(scanLoadIndex, scan);
            }
            scanLoadIndex += lastInvocationInLevel + 1u;
        }
        if(lastInvocationInLevel >= glsl::gl_SubgroupSize())
        {
            if(participate)
            {
                if(any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, isLastSubgroupInvocation))) {
                    sharedAccessor.main.set(scanLoadIndex - loadStoreIndexDiff, scan);
                }
            }
            sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
            participate = gl_LocalInvocationIndex <= (lastInvocationInLevel >>= glsl::gl_SubgroupSizeLog2());
            if(participate)
            {
                const uint prevLevelScan = sharedAccessor.main.get(scanLoadIndex);
                scan = subgroupOp(prevLevelScan);
            }
        }
        lastLevelScan = scan; // only invocations of gl_LocalInvocationIndex < gl_SubgroupSize will have correct values, rest will have garbage
    }
};

template<typename T, class Binop, class ScanHead, class SharedAccessor, bool isExclusive>
struct WorkgroupScanTail
{
    T identity;
    ScanHead head;

    static WorkgroupScanTail create(T identity, ScanHead head)
    {
        WorkgroupScanTail<T, Binop, ScanHead, SharedAccessor, isExclusive> tail;
        tail.identity = identity;
        tail.head = head;
        return tail;
    }

    T operator()(NBL_REF_ARG(SharedAccessor) sharedAccessor)
    {
        Binop binop;
        uint lastInvocation = head.lastInvocation;
        uint firstLevelScan = head.firstLevelScan;
        const uint subgroupId = glsl::gl_SubgroupID();
        
        if(lastInvocation >= glsl::gl_SubgroupSize())
        {
            uint scanLoadIndex = head.scanLoadIndex + glsl::gl_SubgroupSize();
            const uint shiftedInvocationIndex = gl_LocalInvocationIndex + glsl::gl_SubgroupSize();
            const uint currentToHighLevel = subgroupId - shiftedInvocationIndex;
            
            sharedAccessor.main.set(head.scanLoadIndex, head.lastLevelScan);

            for(uint logShift = (firstbithigh(lastInvocation) / glsl::gl_SubgroupSizeLog2() - 1u) * glsl::gl_SubgroupSizeLog2(); logShift > 0u; logShift -= glsl::gl_SubgroupSizeLog2())
            {
                uint lastInvocationInLevel = lastInvocation >> logShift;
                sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
                const uint currentLevelIndex = scanLoadIndex - (lastInvocationInLevel + 1u);
                if(shiftedInvocationIndex <= lastInvocationInLevel)
                {
                    sharedAccessor.main.set(currentLevelIndex, binop(sharedAccessor.main.get(scanLoadIndex+currentToHighLevel), sharedAccessor.main.get(currentLevelIndex)));
                }
                scanLoadIndex = currentLevelIndex;
            }
            sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
            
            if(gl_LocalInvocationIndex <= lastInvocation && subgroupId != 0u)
            {
                const uint higherLevelExclusive = sharedAccessor.main.get(subgroupId - 1u);
                firstLevelScan = binop(higherLevelExclusive, firstLevelScan);
            }
        }
        
        if(isExclusive)
        {
            sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
            firstLevelScan = glsl::subgroupShuffleUp(firstLevelScan, 1u);
            if(glsl::subgroupElect())
            { // shuffle doesn't work between subgroups but the value for each elected subgroup invocation is just the previous higherLevelExclusive
                firstLevelScan = all(bool2(gl_LocalInvocationIndex != 0u, gl_LocalInvocationIndex <= lastInvocation)) ? sharedAccessor.main.get(subgroupId - 1u) : identity;
            }
            return firstLevelScan;
        }
        else
        {
            return firstLevelScan;
        }
    }
};
}
}
}

#endif