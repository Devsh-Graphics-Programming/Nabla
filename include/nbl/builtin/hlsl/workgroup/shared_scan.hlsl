// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_

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
    bool isScan;
    T identity;
    T firstLevelScan;
    uint itemCount;
    uint lastInvocation;
    uint lastInvocationInLevel;
    uint scanStoreIndex;

    static WorkgroupScanHead create(bool isScan, T identity, uint itemCount)
    {
        WorkgroupScanHead<T, SubgroupOp, SharedAccessor> wsh;
        wsh.isScan = isScan;
        wsh.identity = identity;
        wsh.itemCount = itemCount;
        wsh.lastInvocation = itemCount - 1u;
        return wsh;
    }

    T operator()(T value)
    {
        const uint subgroupMask = glsl::gl_SubgroupSize() - 1u;
        SharedAccessor sharedAccessor;
        lastInvocationInLevel = lastInvocation;
        
        SubgroupOp subgroupOp;
        firstLevelScan = subgroupOp(value);
        T scan = firstLevelScan;
        
        const bool isLastSubgroupInvocation = glsl::gl_SubgroupInvocationID() == glsl::gl_SubgroupSize() - 1u;
        
        // Since we are scanning the RESULT of the initial scan (which paired one input per subgroup invocation) 
        // every group of 64 invocations has been coallesced into 1 result value. This means that the results of 
        // the first SubgroupSz^2 invocations will be processed by the first subgroup and so on.
        // Consequently, those first SubgroupSz^2 invocations will store their results on SubgroupSz scratch slots 
        // with halfSubgroupSz padding and the next level will follow the same + the previous as an `offset`.
        const uint offset = (gl_LocalInvocationIndex >> glsl::gl_SubgroupSizeLog2()) & ~subgroupMask; // For subgroupSz = 64, first 4095 invocations get offset 0, 4096-8191 offset 64, then 128 etc.
        const uint memBegin = offset;
        uint nextLevelStoreIndex = memBegin + glsl::gl_SubgroupID();
        scanStoreIndex = gl_LocalInvocationIndex;
        
        bool participate = gl_LocalInvocationIndex <= lastInvocationInLevel;
        while(lastInvocationInLevel >= glsl::gl_SubgroupSize() * glsl::gl_SubgroupSize())
        {
            sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
            if(participate)
            {
                if (any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, isLastSubgroupInvocation)))
                { // only invocations that have the final value of the subgroupOp (inclusive scan) store their results
                    sharedAccessor.main.set(nextLevelStoreIndex, scan);
                }
            }
            sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
            participate = gl_LocalInvocationIndex <= (lastInvocationInLevel >>= glsl::gl_SubgroupSizeLog2());
            if(participate)
            {
                const uint prevLevelScan = sharedAccessor.main.get(gl_LocalInvocationIndex);
                scan = subgroupOp(prevLevelScan);
                if(isScan)
                {
                    sharedAccessor.main.set(scanStoreIndex, scan);
                }
            }
            if(isScan)
            {
                scanStoreIndex += lastInvocationInLevel + 1u;
            }
        }
        if(lastInvocationInLevel >= glsl::gl_SubgroupSize())
        {
            sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
            if(participate)
            {
                if(any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, isLastSubgroupInvocation))) {
                    sharedAccessor.main.set(nextLevelStoreIndex, scan);
                }
            }
            sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
            participate = gl_LocalInvocationIndex <= (lastInvocationInLevel >>= glsl::gl_SubgroupSizeLog2());
            if(participate)
            {
                const uint prevLevelScan = sharedAccessor.main.get(gl_LocalInvocationIndex);
                scan = subgroupOp(prevLevelScan);
                if(isScan) {
                    sharedAccessor.main.set(scanStoreIndex, scan);
                }
            }
        }
        return scan;
    }
};

template<typename T, class Binop, class SharedAccessor>
struct WorkgroupScanTail
{
    bool isExclusive;
    T identity;
    T firstLevelScan;
    uint lastInvocation;
    uint scanStoreIndex;

    static WorkgroupScanTail create(bool isExclusive, T identity, T firstLevelScan, uint lastInvocation, uint scanStoreIndex)
    {
        WorkgroupScanTail<T, Binop, SharedAccessor> wst;
        wst.isExclusive = isExclusive;
        wst.identity = identity;
        wst.firstLevelScan = firstLevelScan;
        wst.lastInvocation = lastInvocation;
        wst.scanStoreIndex = scanStoreIndex;
        return wst;
    }

    T operator()()
    {
        Binop binop;
        SharedAccessor sharedAccessor;
        sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
        
        const uint subgroupId = glsl::gl_SubgroupID();
        if(lastInvocation >= glsl::gl_SubgroupSize())
        {
            uint scanLoadIndex = scanStoreIndex + glsl::gl_SubgroupSize();
            const uint shiftedInvocationIndex = gl_LocalInvocationIndex + glsl::gl_SubgroupSize();
            const uint currentToHighLevel = subgroupId - shiftedInvocationIndex;
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
                firstLevelScan = sharedAccessor.main.get(subgroupId - 1u);
            }
            return all(bool2(gl_LocalInvocationIndex != 0u, gl_LocalInvocationIndex <= lastInvocation)) ? firstLevelScan : identity;
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