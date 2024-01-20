// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

namespace impl
{
template<class BinOp, uint16_t ItemCount, class device_capabilities>
struct reduce
{
    using type_t = typename BinOp::type_t;
    
    template<class Accessor>
    void __call(NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) scratchAccessor)
    {
        const uint16_t lastInvocation = ItemCount-1;
        const uint16_t subgroupMask = uint16_t(glsl::gl_SubgroupSize()-1u);

        subgroup::inclusive_scan<BinOp,device_capabilities> subgroupOp;

        lastInvocationInLevel = lastInvocation;
        scanLoadIndex = SubgroupContiguousIndex();
        participate = scanLoadIndex<=lastInvocationInLevel;

        firstLevelScan = subgroupOp(participate ? value:BinOp::identity);
        type_t scan = firstLevelScan;
        
        // could use ElectLast() but we can optimize for full workgroups here
        const bool isLastSubgroupInvocation = uint16_t(glsl::gl_SubgroupInvocationID())==subgroupMask;
        
        // Since we are scanning the RESULT of the initial scan (which paired one input per subgroup invocation) 
        // every group of gl_SubgroupSz invocations has been coallesced into 1 result value. This means that the results of 
        // the first gl_SubgroupSz^2 invocations will be processed by the first subgroup and so on.
        // Consequently, those first gl_SubgroupSz^2 invocations will store their results on gl_SubgroupSz scratch slots 
        // and the next level will follow the same + the previous as an `offset`.
        
        const uint16_t loadStoreIndexDiff = scanLoadIndex-uint16_t(glsl::gl_SubgroupID());
        
        // to cancel out the index shift on the first iteration
        if (lastInvocationInLevel>subgroupMask)
             scanLoadIndex -= lastInvocationInLevel-1;
        // TODO: later [unroll(scan_levels<ItemCount,subgroup::MinSubgroupSize>::value-1)]
        [unroll(1)]
        while (lastInvocationInLevel>subgroupMask)
        {
            scanLoadIndex += lastInvocationInLevel+1;
            // only invocations that have the final value of the subgroupOp (inclusive scan) store their results
            if (participate && (SubgroupContiguousIndex()==lastInvocationInLevel || isLastSubgroupInvocation))
                scratchAccessor.set(scanLoadIndex-loadStoreIndexDiff,scan); // For subgroupSz = 32, first 512 invocations store index is [0,15], 512-1023 [16,31] etc.
            scratchAccessor.workgroupExecutionAndMemoryBarrier();
            participate = SubgroupContiguousIndex() <= (lastInvocationInLevel >>= glsl::gl_SubgroupSizeLog2());
            if(participate)
            {
                const type_t prevLevelScan = scratchAccessor.get(scanLoadIndex);
                scan = subgroupOp(prevLevelScan);
            }
        }
        lastLevelScan = scan; // only invocations of SubgroupContiguousIndex() < gl_SubgroupSize will have correct values, rest will have garbage
    }

    type_t firstLevelScan;
    type_t lastLevelScan;
    uint16_t lastInvocationInLevel;
    uint16_t scanLoadIndex;
    bool participate;
};

template<class BinOp, bool Exclusive, uint16_t ItemCount, class device_capabilities>
struct scan// : reduce<BinOp,ItemCount> https://github.com/microsoft/DirectXShaderCompiler/issues/5966
{
    using base_t = reduce<BinOp,ItemCount,device_capabilities>;
    base_t __base;
    using type_t = typename base_t::type_t;

    template<class Accessor>
    type_t __call(NBL_CONST_REF_ARG(type_t) value, NBL_REF_ARG(Accessor) scratchAccessor)
    {
        __base.template __call<Accessor>(value,scratchAccessor);
        
        const uint16_t subgroupID = uint16_t(glsl::gl_SubgroupID());
        // abuse integer wraparound to map 0 to 0xffffu
        const uint16_t prevSubgroupID = subgroupID-1;
        
        // important check to prevent weird `firstbithigh` overlflows
        const uint16_t lastInvocation = ItemCount-1;
        if(lastInvocation>=uint16_t(glsl::gl_SubgroupSize()))
        {
            const uint16_t subgroupSizeLog2 = uint16_t(glsl::gl_SubgroupSizeLog2());
            // different than Upsweep cause we need to translate high level inclusive scans into exclusive on the fly, so we get the value of the subgroup behind our own in each level
            const uint16_t storeLoadIndexDiff = SubgroupContiguousIndex()-prevSubgroupID;
            
            BinOp binop;
            // because DXC doesn't do references and I need my "frozen" registers
            #define scanStoreIndex __base.scanLoadIndex
            // we sloop over levels from highest to penultimate
            // as we iterate some previously active (higher level) invocations hold their exclusive prefix sum in `lastLevelScan`
            const uint16_t temp = uint16_t(firstbithigh(uint32_t(lastInvocation))/subgroupSizeLog2); // doing division then multiplication might be optimized away by the compiler
            const uint16_t initialLogShift = temp*subgroupSizeLog2;
            // TODO: later [unroll(scan_levels<ItemCount,MinSubgroupSize>::value-1)]
            [unroll(1)]
            for (uint16_t logShift=initialLogShift; bool(logShift); logShift-=subgroupSizeLog2)
            {
                // on the first iteration gl_SubgroupID==0 will participate but not afterwards because binop operand is identity
                if (__base.participate)
                {
                    // we need to add the higher level invocation exclusive prefix sum to current value
                    if (logShift!=initialLogShift) // but the top level doesn't have any level above itself
                    {
                        // this is fine if on the way up you also += under `if (participate)`
                        scanStoreIndex -= __base.lastInvocationInLevel+1;
                        __base.lastLevelScan = binop(__base.lastLevelScan,scratchAccessor.get(scanStoreIndex));
                    }
                    // now `lastLevelScan` has current level's inclusive prefux sum computed properly
                    // note we're overwriting the same location with same invocation so no barrier needed
                    // we store everything even though we'll never use the last entry due to shuffleup on read
                    scratchAccessor.set(scanStoreIndex,__base.lastLevelScan);
                }
                scratchAccessor.workgroupExecutionAndMemoryBarrier();
                // we're sneaky and exclude `gl_SubgroupID==0`  from participation by abusing integer underflow
                __base.participate = prevSubgroupID<__base.lastInvocationInLevel;
                if (__base.participate)
                {
                    // we either need to prevent OOB read altogether OR cmov identity after the far
                    __base.lastLevelScan = scratchAccessor.get(scanStoreIndex-storeLoadIndexDiff);
                }
                __base.lastInvocationInLevel = lastInvocation>>logShift;
            }
            #undef scanStoreIndex
            
            //assert((__base.lastInvocation>>subgroupSizeLog2)==__base.lastInvocationInLevel);
            
            // the very first prefix sum we did is in a register, not Accessor scratch mem hence the special path
            if (prevSubgroupID<__base.lastInvocationInLevel)
                __base.firstLevelScan = binop(__base.lastLevelScan,__base.firstLevelScan);
        }
        
        if(Exclusive)
        {
            __base.firstLevelScan = glsl::subgroupShuffleUp(__base.firstLevelScan,1);
            // shuffle doesn't work between subgroups but the value for each elected subgroup invocation is just the previous higherLevelExclusive
            // note that we assume we might have to do scans with itemCount <= gl_WorkgroupSize
            if (glsl::subgroupElect())
                __base.firstLevelScan = bool(subgroupID) ? __base.lastLevelScan:BinOp::identity;
        }
        return __base.firstLevelScan;
    }
};
}

}
}
}

#endif