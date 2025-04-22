// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_SCAN_DEFAULT_SCHEDULER_INCLUDED_
#define _NBL_HLSL_SCAN_DEFAULT_SCHEDULER_INCLUDED_

#include "nbl/builtin/hlsl/scan/declarations.hlsl"
#include "nbl/builtin/hlsl/scan/descriptors.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{
namespace scheduler
{
    /**
     * The CScanner.h parameter computation calculates the number of virtual workgroups that will have to be launched for the Scan operation
     * (always based on the elementCount) as well as different offsets for the results of each step of the Scan operation, flag positions 
     * that are used for synchronization etc.
     * Remember that CScanner does a Blelloch Scan which works in levels. In each level of the Blelloch scan the array of elements is 
     * broken down into sets of size=WorkgroupSize and each set is scanned using Hillis & Steele (aka Stone-Kogge adder). The result of 
     * the scan is provided as an array element for the next level of the Blelloch Scan. This means that if we have 10000 elements and 
     * WorkgroupSize=250, we will break the array into 40 sets and take their reduction results. The next level of the Blelloch Scan will 
     * have an array of size 40. Only a single workgroup will be needed to work on that. After that array is scanned, we use the results 
     * in the downsweep phase of Blelloch Scan.
     * Keep in mind that each virtual workgroup executes a single step of the whole algorithm, which is why we have the cumulativeWorkgroupCount.
     * The first virtual workgroups will work on the upsweep phase, the next on the downsweep phase.
     * The intermediate results are stored in a scratch buffer. That buffer's size is is the sum of the element-array size for all the 
     * Blelloch levels. Using the previous example, the scratch size should be 10000 + 40.
     * 
     * Parameter meaning:
     * |> lastElement - the index of the last element of each Blelloch level in the scratch buffer
     * |> topLevel - the top level the Blelloch Scan will have (this depends on the elementCount and the WorkgroupSize)
     * |> temporaryStorageOffset - an offset array for each level of the Blelloch Scan. It is used when storing the REDUCTION result of each workgroup scan
     * |> cumulativeWorkgroupCount - the sum-scan of all the workgroups that will need to be launched for each level of the Blelloch Scan (both upsweep and downsweep)
     * |> finishedFlagOffset - an index in the scratch buffer where each virtual workgroup indicates that ALL its invocations have finished their work. This helps 
     *                            synchronizing between workgroups with while-loop spinning.
     */
    void computeParameters(NBL_CONST_REF_ARG(uint32_t) elementCount, out Parameters_t _scanParams, out DefaultSchedulerParameters_t _schedulerParams)
    {
#define WorkgroupCount(Level) (_scanParams.lastElement[Level+1]+1u)
        const uint32_t workgroupSizeLog2 = firstbithigh(glsl::gl_WorkGroupSize().x);
        _scanParams.lastElement[0] = elementCount-1u;
        _scanParams.topLevel = firstbithigh(_scanParams.lastElement[0]) / workgroupSizeLog2;
        
        for (uint32_t i=0; i<NBL_BUILTIN_MAX_LEVELS/2;)
        {
            const uint32_t next = i+1;
            _scanParams.lastElement[next] = _scanParams.lastElement[i]>>workgroupSizeLog2;
            i = next;
        }
        _schedulerParams.cumulativeWorkgroupCount[0] = WorkgroupCount(0);
        _schedulerParams.finishedFlagOffset[0] = 0u;
        switch(_scanParams.topLevel)
        {
            case 1u:
                _schedulerParams.cumulativeWorkgroupCount[1] = _schedulerParams.cumulativeWorkgroupCount[0]+1u;
                _schedulerParams.cumulativeWorkgroupCount[2] = _schedulerParams.cumulativeWorkgroupCount[1]+WorkgroupCount(0);
                // climb up
                _schedulerParams.finishedFlagOffset[1] = 1u;
                
                _scanParams.temporaryStorageOffset[0] = 2u;
                break;
            case 2u:
                _schedulerParams.cumulativeWorkgroupCount[1] = _schedulerParams.cumulativeWorkgroupCount[0]+WorkgroupCount(1);
                _schedulerParams.cumulativeWorkgroupCount[2] = _schedulerParams.cumulativeWorkgroupCount[1]+1u;
                _schedulerParams.cumulativeWorkgroupCount[3] = _schedulerParams.cumulativeWorkgroupCount[2]+WorkgroupCount(1);
                _schedulerParams.cumulativeWorkgroupCount[4] = _schedulerParams.cumulativeWorkgroupCount[3]+WorkgroupCount(0);
                // climb up
                _schedulerParams.finishedFlagOffset[1] = WorkgroupCount(1);
                _schedulerParams.finishedFlagOffset[2] = _schedulerParams.finishedFlagOffset[1]+1u;
                // climb down
                _schedulerParams.finishedFlagOffset[3] = _schedulerParams.finishedFlagOffset[1]+2u;
                
                _scanParams.temporaryStorageOffset[0] = _schedulerParams.finishedFlagOffset[3]+WorkgroupCount(1);
                _scanParams.temporaryStorageOffset[1] = _scanParams.temporaryStorageOffset[0]+WorkgroupCount(0);
                break;
            case 3u:
                _schedulerParams.cumulativeWorkgroupCount[1] = _schedulerParams.cumulativeWorkgroupCount[0]+WorkgroupCount(1);
                _schedulerParams.cumulativeWorkgroupCount[2] = _schedulerParams.cumulativeWorkgroupCount[1]+WorkgroupCount(2);
                _schedulerParams.cumulativeWorkgroupCount[3] = _schedulerParams.cumulativeWorkgroupCount[2]+1u;
                _schedulerParams.cumulativeWorkgroupCount[4] = _schedulerParams.cumulativeWorkgroupCount[3]+WorkgroupCount(2);
                _schedulerParams.cumulativeWorkgroupCount[5] = _schedulerParams.cumulativeWorkgroupCount[4]+WorkgroupCount(1);
                _schedulerParams.cumulativeWorkgroupCount[6] = _schedulerParams.cumulativeWorkgroupCount[5]+WorkgroupCount(0);
                // climb up
                _schedulerParams.finishedFlagOffset[1] = WorkgroupCount(1);
                _schedulerParams.finishedFlagOffset[2] = _schedulerParams.finishedFlagOffset[1]+WorkgroupCount(2);
                _schedulerParams.finishedFlagOffset[3] = _schedulerParams.finishedFlagOffset[2]+1u;
                // climb down
                _schedulerParams.finishedFlagOffset[4] = _schedulerParams.finishedFlagOffset[2]+2u;
                _schedulerParams.finishedFlagOffset[5] = _schedulerParams.finishedFlagOffset[4]+WorkgroupCount(2);
                
                _scanParams.temporaryStorageOffset[0] = _schedulerParams.finishedFlagOffset[5]+WorkgroupCount(1);
                _scanParams.temporaryStorageOffset[1] = _scanParams.temporaryStorageOffset[0]+WorkgroupCount(0);
                _scanParams.temporaryStorageOffset[2] = _scanParams.temporaryStorageOffset[1]+WorkgroupCount(1);
                break;
            default:
                break;
#if NBL_BUILTIN_MAX_LEVELS>7
#error "Switch needs more cases"
#endif
        }
#undef WorkgroupCount
    }
    
    /**
     * treeLevel - the current level in the Blelloch Scan
     * localWorkgroupIndex - the workgroup index the current invocation is a part of in the specific virtual dispatch. 
     * For example, if we have dispatched 10 workgroups and we the virtu    al workgroup number is 35, then the localWorkgroupIndex should be 5.
     */
    template<class Accessor>
    bool getWork(NBL_CONST_REF_ARG(DefaultSchedulerParameters_t) params, NBL_CONST_REF_ARG(uint32_t) topLevel, NBL_REF_ARG(uint32_t) treeLevel, NBL_REF_ARG(uint32_t) localWorkgroupIndex, NBL_REF_ARG(Accessor) sharedScratch)
    {
        const uint32_t lastLevel = topLevel<<1u;
		if(workgroup::SubgroupContiguousIndex() == 0u) 
		{
		  uint32_t originalWGs = glsl::atomicAdd(scanScratchBuf[0].workgroupsStarted, 1u);
		  sharedScratch.set(0u, originalWGs);
		  treeLevel = 0;
		  while (originalWGs>=params.cumulativeWorkgroupCount[treeLevel]) { // doesn't work for now because PushConstant arrays can only be accessed by dynamically uniform indices
			treeLevel++;
		  }
		  sharedScratch.set(1u, treeLevel);
		}
		sharedScratch.workgroupExecutionAndMemoryBarrier();
		const uint32_t globalWorkgroupIndex = sharedScratch.get(0u);
        
        treeLevel = sharedScratch.get(1u);
        if(treeLevel>lastLevel)
            return true;
        
        localWorkgroupIndex = globalWorkgroupIndex;
        const bool dependentLevel = treeLevel != 0u;
        if(dependentLevel) 
        {
            const uint32_t prevLevel = treeLevel - 1u;
            localWorkgroupIndex -= params.cumulativeWorkgroupCount[prevLevel];
            if(workgroup::SubgroupContiguousIndex() == 0u) 
            {
                uint32_t dependentsCount = 1u;
                if(treeLevel <= topLevel) 
                {
                    dependentsCount = glsl::gl_WorkGroupSize().x;
                    const bool lastWorkgroup = (globalWorkgroupIndex+1u)==params.cumulativeWorkgroupCount[treeLevel];
                    if (lastWorkgroup) 
                    {
                        const Parameters_t scanParams = getParameters();
                        dependentsCount = scanParams.lastElement[treeLevel]+1u;
                        if (treeLevel<topLevel) 
                        {
                            dependentsCount -= scanParams.lastElement[treeLevel+1u] * glsl::gl_WorkGroupSize().x;
                        }
                    }
                }
                uint32_t dependentsFinishedFlagOffset = localWorkgroupIndex;
                if (treeLevel>topLevel) // !(prevLevel<topLevel) TODO: merge with `else` above?
                    dependentsFinishedFlagOffset /= glsl::gl_WorkGroupSize().x;
                dependentsFinishedFlagOffset += params.finishedFlagOffset[prevLevel];
                while (scanScratchBuf[0].data[dependentsFinishedFlagOffset]!=dependentsCount)
                    glsl::memoryBarrierBuffer();
            }
        }
        sharedScratch.workgroupExecutionAndMemoryBarrier();
        glsl::memoryBarrierBuffer();
        return false;
    }
    
    void markComplete(NBL_CONST_REF_ARG(DefaultSchedulerParameters_t) params, NBL_CONST_REF_ARG(uint32_t) topLevel, NBL_CONST_REF_ARG(uint32_t) treeLevel, NBL_CONST_REF_ARG(uint32_t) localWorkgroupIndex)
    {
        glsl::memoryBarrierBuffer();
        if (workgroup::SubgroupContiguousIndex()==0u)
        {
            uint32_t finishedFlagOffset = params.finishedFlagOffset[treeLevel];
            if (treeLevel<topLevel)
            {
                finishedFlagOffset += localWorkgroupIndex/glsl::gl_WorkGroupSize().x;
                glsl::atomicAdd(scanScratchBuf[0].data[finishedFlagOffset], 1u);
            }
            else if (treeLevel!=(topLevel<<1u))
            {
                finishedFlagOffset += localWorkgroupIndex;
                scanScratchBuf[0].data[finishedFlagOffset] = 1u;
            }
        }
    }
}

}
}
}


#endif