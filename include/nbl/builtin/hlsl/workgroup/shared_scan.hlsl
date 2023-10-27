// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

template<typename T, class SubgroupOp, class SharedAccessor, uint itemCount>
struct Reduce
{
    T firstLevelScan;
    T lastLevelScan;
    uint lastInvocation;
    uint lastInvocationInLevel;
    uint scanLoadIndex;
    bool participate;

    static Reduce create()
    {
        Reduce<T, SubgroupOp, SharedAccessor, itemCount> wsh;
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
        // every group of gl_SubgroupSz invocations has been coallesced into 1 result value. This means that the results of 
        // the first gl_SubgroupSz^2 invocations will be processed by the first subgroup and so on.
        // Consequently, those first gl_SubgroupSz^2 invocations will store their results on gl_SubgroupSz scratch slots 
        // and the next level will follow the same + the previous as an `offset`.
        
        scanLoadIndex = SubgroupContiguousIndex();
        const uint loadStoreIndexDiff = scanLoadIndex - glsl::gl_SubgroupID();
        
        participate = SubgroupContiguousIndex() <= lastInvocationInLevel;
        // to cancel out the index shift on the first iteration
        if (lastInvocationInLevel >= glsl::gl_SubgroupSize())
             scanLoadIndex -= lastInvocationInLevel-1u;
        // TODO: later [unroll(scan_levels<WorkgroupSize,MinSubgroupSize>::value-1)]
        [unroll(1)]
        while(lastInvocationInLevel >= glsl::gl_SubgroupSize())
        {
            scanLoadIndex += lastInvocationInLevel+1u;
            // only invocations that have the final value of the subgroupOp (inclusive scan) store their results
            if (participate && (SubgroupContiguousIndex()==lastInvocationInLevel || isLastSubgroupInvocation))
                sharedAccessor.main.set(scanLoadIndex - loadStoreIndexDiff, scan); // For subgroupSz = 32, first 512 invocations store index is [0,15], 512-1023 [16,31] etc.
            sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
            participate = SubgroupContiguousIndex() <= (lastInvocationInLevel >>= glsl::gl_SubgroupSizeLog2());
            if(participate)
            {
                const uint prevLevelScan = sharedAccessor.main.get(scanLoadIndex);
                scan = subgroupOp(prevLevelScan);
            }
        }
        lastLevelScan = scan; // only invocations of SubgroupContiguousIndex() < gl_SubgroupSize will have correct values, rest will have garbage
    }
};

template<typename T, class Binop, class SubgroupScanOp, class SharedAccessor, uint itemCount, bool isExclusive>
struct Scan
{
    Reduce<T, SubgroupScanOp, SharedAccessor, itemCount> reduce;

    static Scan create()
    {
        Scan<T, Binop, SubgroupScanOp, SharedAccessor, itemCount, isExclusive> scan;
        scan.reduce = Reduce<T, SubgroupScanOp, SharedAccessor, itemCount>::create();
        return scan;
    }

    T operator()(T value, NBL_REF_ARG(SharedAccessor) sharedAccessor)
    {
        reduce(value, sharedAccessor);
        
        Binop binop;
        uint lastInvocation = reduce.lastInvocation;
        uint firstLevelScan = reduce.firstLevelScan;
        const uint subgroupId = glsl::gl_SubgroupID();
        
        // abuse integer wraparound to map 0 to 0xffFFffFFu
        const uint32_t prevSubgroupID = uint32_t(glsl::gl_SubgroupID())-1u;
        
        // important check to prevent weird `firstbithigh` overlflows
        if(lastInvocation >= glsl::gl_SubgroupSize())
        {
            // different than Upsweep cause we need to translate high level inclusive scans into exclusive on the fly, so we get the value of the subgroup behind our own in each level
            const uint32_t storeLoadIndexDiff = uint32_t(SubgroupContiguousIndex()) - prevSubgroupID ;
            
            // because DXC doesn't do references and I need my "frozen" registers
            #define scanStoreIndex reduce.scanLoadIndex
            // we sloop over levels from highest to penultimate
            // as we iterate some previously active (higher level) invocations hold their exclusive prefix sum in `lastLevelScan`
            const uint32_t temp = firstbithigh(lastInvocation) / glsl::gl_SubgroupSizeLog2(); // doing division then multiplication might be optimized away by the compiler
            const uint32_t initialLogShift = temp * glsl::gl_SubgroupSizeLog2();
            // TODO: later [unroll(scan_levels<WorkgroupSize,MinSubgroupSize>::value-1)]
            [unroll(1)]
            for(uint32_t logShift=initialLogShift; bool(logShift); logShift-=glsl::gl_SubgroupSizeLog2())
            {
                // on the first iteration gl_SubgroupID==0 will participate but not afterwards because binop operand is identity
                if (reduce.participate)
                {
                    // we need to add the higher level invocation exclusive prefix sum to current value
                    if (logShift!=initialLogShift) // but the top level doesn't have any level above itself
                    {
                        // this is fine if on the way up you also += under `if (participate)`
                        scanStoreIndex -= reduce.lastInvocationInLevel+1;
                        reduce.lastLevelScan = binop(reduce.lastLevelScan,sharedAccessor.main.get(scanStoreIndex));
                    }
                    // now `lastLevelScan` has current level's inclusive prefux sum computed properly
                    // note we're overwriting the same location with same invocation so no barrier needed
                    // we store everything even though we'll never use the last entry due to shuffleup on read
                    sharedAccessor.main.set(scanStoreIndex,reduce.lastLevelScan);
                }
                sharedAccessor.main.workgroupExecutionAndMemoryBarrier();
                // we're sneaky and exclude `gl_SubgroupID==0`  from participation by abusing integer underflow
                reduce.participate = prevSubgroupID<reduce.lastInvocationInLevel;
                if (reduce.participate)
                {
                    // we either need to prevent OOB read altogether OR cmov identity after the far
                    reduce.lastLevelScan = sharedAccessor.main.get(scanStoreIndex-storeLoadIndexDiff);
                }
                reduce.lastInvocationInLevel = lastInvocation >> logShift;
            }
            #undef scanStoreIndex
            
            //assert((lastInvocation>>glsl::gl_SubgroupSizeLog2())==reduce.lastInvocationInLevel);
            
            // the very first prefix sum we did is in a register, not Accessor scratch mem hence the special path
            if ( prevSubgroupID < reduce.lastInvocationInLevel)
                firstLevelScan = binop(reduce.lastLevelScan,firstLevelScan);
        }
        
        if(isExclusive)
        {
            firstLevelScan = glsl::subgroupShuffleUp(firstLevelScan, 1u);
            if(glsl::subgroupElect())
            {   // shuffle doesn't work between subgroups but the value for each elected subgroup invocation is just the previous higherLevelExclusive
                // note that we assume we might have to do scans with itemCount <= gl_WorkgroupSize
                firstLevelScan = bool(subgroupId) ? reduce.lastLevelScan : Binop::identity();
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