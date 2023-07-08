// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/scratch.hlsl"

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
        SubgroupOp subgroupOp;
        firstLevelScan = subgroupOp(value);
        T scan = firstLevelScan;
        const bool isLastSubgroupInvocation = offsetsAndMasks.subgroupInvocation == offsetsAndMasks.subgroupMask; // last invocation in subgroup
        const uint subgroupId = subgroup::SubgroupID();
        //const uint nextLevelStoreIndex = offsetsAndMasks.subgroupMemoryBegin + offsetsAndMasks.halfSubgroupSize + subgroupId; // Still wrong, it needs to be subgroupId + halfSubgroupSize + subgroupMemBegin(subgroupId) NOT the subgrMemBegin of the localinvocationidx
        const uint nextLevelStoreIndex = offsetsAndMasks.halfSubgroupSize + subgroupId;
        // REVIEW: use broadcast instead of getSubgroupEmulationMemoryStoreOffset(loMask,lastInvocation)
        uint scanStoreIndex = subgroup::Broadcast(offsetsAndMasks.scanStoreOffset, lastInvocation) + gl_LocalInvocationIndex + 1u;
        bool participate = gl_LocalInvocationIndex <= lastInvocationInLevel;
		//if(gl_WorkGroupID.x==0 && isLastSubgroupInvocation)
			//printf("LocalInvoc %u SubgrId %u NextLvlStore %u ScanStoreIdx %u\n", gl_LocalInvocationIndex, subgroupId, nextLevelStoreIndex, scanStoreIndex);
        while(lastInvocationInLevel >= subgroup::Size() * subgroup::Size())
    	{
			if(gl_WorkGroupID.x==0)
				printf("SHOULDN'T GO IN HERE\n");
    		Barrier();
    		if(participate)
    		{
    			if (any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, isLastSubgroupInvocation)))
    			{
    				scratch.set(nextLevelStoreIndex, scan);
    			}
    		}
    		Barrier();
    		participate = gl_LocalInvocationIndex <= (lastInvocationInLevel >>= subgroup::SizeLog2());
    		if(participate)
    		{
				// TODO (PentaKon): Not sure all of this scratch assigning is needed, test if we can just use the `scan` variable everywhere
    			const uint prevLevelScan = scratch.get(offsetsAndMasks.scanStoreOffset);
    			scan = subgroupOp(prevLevelScan);
    			if(isScan)
    				scratch.set(scanStoreIndex, scan);
    		}
    		if(isScan)
    			scanStoreIndex += lastInvocationInLevel + 1u;
    	}
    	if(lastInvocationInLevel >= subgroup::Size())
    	{
    		Barrier();
			// TODO (PentaKon): same as above, maybe we can just the `scan` and not do anything with scratch?
    		if(participate)
    		{
    			if(any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, isLastSubgroupInvocation)))
    				scratch.set(nextLevelStoreIndex, scan);
    		}
    		Barrier();
    		participate = gl_LocalInvocationIndex <= (lastInvocationInLevel >>= subgroup::SizeLog2());
    		if(participate)
    		{
    			const uint prevLevelScan = scratch.get(offsetsAndMasks.scanStoreOffset);
    			scan = subgroupOp(prevLevelScan);
    			if(isScan)
    				scratch.set(scanStoreIndex, scan);
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
        subgroup::ScratchOffsetsAndMasks offsetsAndMasks;

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
    				scratch.set(currentLevelIndex, binop(scratch.get(scanLoadIndex+currentToHighLevel), scratch.get(currentLevelIndex)));
    			}
    			scanLoadIndex = currentLevelIndex;
    		}
    		Barrier();
    		if(gl_LocalInvocationIndex <= lastInvocation && subgroupId != 0u)
    		{
    			const uint higherLevelExclusive = scratch.get(scanLoadIndex + currentToHighLevel - 1u);
    			firstLevelScan = binop(higherLevelExclusive, firstLevelScan);
    		}
    	}
    	if(isExclusive)
    	{
    		if(gl_LocalInvocationIndex < lastInvocation)
    		{
    			scratch.set(gl_LocalInvocationIndex + 1u, firstLevelScan);
    		}
    		Barrier();
    		return any(bool2(gl_LocalInvocationIndex != 0u, gl_LocalInvocationIndex <= lastInvocation)) ? scratch.get(gl_LocalInvocationIndex) : identity;
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