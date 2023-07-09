// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/scratch.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

template<typename T, class SubgroupOp, class ScratchAccessor>
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
        WorkgroupScanHead<T, SubgroupOp, ScratchAccessor> wsh;
        wsh.isScan = isScan;
        wsh.identity = identity;
        wsh.itemCount = itemCount;
        wsh.lastInvocation = itemCount - 1u;
        return wsh;
    }

    T operator()(T value)
    {
        subgroup::ScratchOffsetsAndMasks offsetsAndMasks = subgroup::ScratchOffsetsAndMasks::WithSubgroupOpDefaults();
        ScratchAccessor scratch;
        subgroup::scratchInitialize<ScratchAccessor, T>(value, identity, itemCount);
        lastInvocationInLevel = lastInvocation;
		scanStoreIndex = Broadcast<T, ScratchAccessor>(offsetsAndMasks.scanStoreOffset, lastInvocation) + gl_LocalInvocationIndex + 1u;
		
        SubgroupOp subgroupOp;
        firstLevelScan = subgroupOp(value);
        T scan = firstLevelScan;
		
        const bool isLastSubgroupInvocation = offsetsAndMasks.subgroupInvocation == offsetsAndMasks.subgroupMask; // last invocation in subgroup
        const uint subgroupId = subgroup::SubgroupID();
        const uint nextLevelStoreIndex = offsetsAndMasks.halfSubgroupSize + subgroupId; // TODO (PentaKon) Fix this, it should be subgroupId + halfSubgroupSize + subgroupMemBegin(subgroupId)
        bool participate = gl_LocalInvocationIndex <= lastInvocationInLevel;
        while(lastInvocationInLevel >= subgroup::Size() * subgroup::Size())
    	{
    		Barrier();
    		if(participate)
    		{
    			if (any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, isLastSubgroupInvocation)))
    			{
    				scratch.main.set(nextLevelStoreIndex, scan);
    			}
    		}
    		Barrier();
    		participate = gl_LocalInvocationIndex <= (lastInvocationInLevel >>= subgroup::SizeLog2());
    		if(participate)
    		{
    			const uint prevLevelScan = scratch.main.get(offsetsAndMasks.scanStoreOffset);
    			scan = subgroupOp(prevLevelScan);
    			if(isScan)
    				scratch.main.set(scanStoreIndex, scan);
    		}
    		if(isScan)
    			scanStoreIndex += lastInvocationInLevel + 1u;
    	}
    	if(lastInvocationInLevel >= subgroup::Size())
    	{
    		Barrier();
    		if(participate)
    		{
    			if(any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, isLastSubgroupInvocation)))
    				scratch.main.set(nextLevelStoreIndex, scan);
    		}
    		Barrier();
    		participate = gl_LocalInvocationIndex <= (lastInvocationInLevel >>= subgroup::SizeLog2());
    		if(participate)
    		{
    			const uint prevLevelScan = scratch.main.get(offsetsAndMasks.scanStoreOffset);
    			scan = subgroupOp(prevLevelScan);
    			if(isScan) {
    				scratch.main.set(scanStoreIndex, scan);
				}
    		}
    	}
    	return scan;
    }
};

template<typename T, class Binop, class ScratchAccessor>
struct WorkgroupScanTail
{
    bool isExclusive;
    T identity;
    T firstLevelScan;
    uint lastInvocation;
    uint scanStoreIndex;

    static WorkgroupScanTail create(bool isExclusive, T identity, T firstLevelScan, uint lastInvocation, uint scanStoreIndex)
    {
        WorkgroupScanTail<T, Binop, ScratchAccessor> wst;
        wst.isExclusive = isExclusive;
        wst.identity = identity;
        wst.firstLevelScan = firstLevelScan;
        wst.lastInvocation = lastInvocation;
        wst.scanStoreIndex = scanStoreIndex;
        return wst;
    }

    T operator()()
    {
		Barrier();
        Binop binop;
        ScratchAccessor scratch;
        subgroup::ScratchOffsetsAndMasks offsetsAndMasks = subgroup::ScratchOffsetsAndMasks::WithSubgroupOpDefaults(); // TODO (PentaKon): REMOVE
		
        if(lastInvocation >= subgroup::Size())
    	{
			const uint subgroupId = subgroup::SubgroupID();
    		uint scanLoadIndex = scanStoreIndex + subgroup::Size();
    		const uint shiftedInvocationIndex = gl_LocalInvocationIndex + subgroup::Size();
    		const uint currentToHighLevel = subgroupId - shiftedInvocationIndex;
    		for(uint logShift = (firstbithigh(lastInvocation) / subgroup::SizeLog2() - 1u) * subgroup::SizeLog2(); logShift > 0u; logShift -= subgroup::SizeLog2())
    		{
    			uint lastInvocationInLevel = lastInvocation >> logShift;
    			Barrier();
    			const uint currentLevelIndex = scanLoadIndex - (lastInvocationInLevel + 1u);
    			if(shiftedInvocationIndex <= lastInvocationInLevel)
    			{
    				scratch.main.set(currentLevelIndex, binop(scratch.main.get(scanLoadIndex+currentToHighLevel), scratch.main.get(currentLevelIndex)));
    			}
    			scanLoadIndex = currentLevelIndex;
    		}
    		Barrier();
			
    		if(gl_LocalInvocationIndex <= lastInvocation && subgroupId != 0u)
    		{
    			const uint higherLevelExclusive = scratch.main.get(scanLoadIndex + currentToHighLevel - 1u);
				//if(offsetsAndMasks.subgroupElectedLocalInvocation == gl_LocalInvocationIndex && offsetsAndMasks.subgroupElectedLocalInvocation != higherLevelExclusive) {
				//	printf("[%u].[%u].[%u] Will be adding %u + %u\n", gl_WorkGroupID.x, gl_LocalInvocationIndex, offsetsAndMasks.subgroupElectedLocalInvocation, higherLevelExclusive, firstLevelScan);
				//}
    			firstLevelScan = binop(higherLevelExclusive, firstLevelScan);
    		}
    	}
		
    	if(isExclusive)
    	{
    		if(gl_LocalInvocationIndex < lastInvocation)
    		{
    			scratch.main.set(gl_LocalInvocationIndex + 1u, firstLevelScan);
    		}
			Barrier();
    		return any(bool2(gl_LocalInvocationIndex != 0u, gl_LocalInvocationIndex <= lastInvocation)) ? scratch.main.get(gl_LocalInvocationIndex) : identity;
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