// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_SHARED_SCAN_INCLUDED_

#ifndef _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
#define _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
const uint gl_LocalInvocationIndex : SV_GroupIndex;
#endif

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
    uint itemCount;
    uint lastInvocation;
    uint lastInvocationInLevel;
    uint scanStoreIndex;

    static WorkgroupScanHead create(bool isScan, T identity, uint itemCount /*bitfieldDWORDs*/)
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
        subgroup::ScratchOffsetsAndMasks offsetsAndMasks = subgroup::ScratchOffsetsAndMasks::WithDefaults();
        ScratchAccessor scratch;
        subgroup::scratchInitialize<ScratchAccessor, T>(value, identity, itemCount);
        lastInvocationInLevel = lastInvocation;
        SubgroupOp subgroupOp;
        T firstLevelScan = subgroupOp(value);
        T scan = firstLevelScan;
        const bool possibleProp = offsetsAndMasks.subgroupInvocation == offsetsAndMasks.subgroupMask; // last invocation in subgroup
        const uint subgroupId = gl_LocalInvocationIndex >> subgroup::SizeLog2();
        const uint nextStoreIndex = offsetsAndMasks.scanStoreOffset;
        // REVIEW: use broadcast instead of getSubgroupEmulationMemoryStoreOffset(loMask,lastInvocation)
        uint scanStoreIndex = subgroup::Broadcast(offsetsAndMasks.scanStoreOffset, lastInvocation) + gl_LocalInvocationIndex + 1u;
        bool participate = gl_LocalInvocationIndex <= lastInvocationInLevel;

        while(lastInvocationInLevel >= subgroup::Size() * subgroup::Size())
    	{
    		Barrier();
    		if(participate)
    		{
    			if (any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, possibleProp)))
    			{
    				scratch.set(nextStoreIndex, scan);
    			}
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
    		if(isScan)
    			scanStoreIndex += lastInvocationInLevel + 1u;
    	}
    	if(lastInvocationInLevel >= subgroup::Size())
    	{
    		Barrier();
    		if(participate)
    		{
    			if(any(bool2(gl_LocalInvocationIndex == lastInvocationInLevel, possibleProp)))
    				scratch.set(nextStoreIndex, scan);
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
    		uint scanLoadIndex = scanStoreIndex + subgroup::Size();
    		const uint shiftedInvocationIndex = gl_LocalInvocationIndex + subgroup::Size();
    		const uint currentToHighLevel = offsetsAndMasks.subgroupInvocation - shiftedInvocationIndex;
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
    		if(gl_LocalInvocationIndex <= lastInvocation && offsetsAndMasks.subgroupInvocation != 0u)
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