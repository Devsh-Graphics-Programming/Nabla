#ifndef _NBL_HLSL_SCAN_DEFAULT_SCHEDULER_INCLUDED_
#define _NBL_HLSL_SCAN_DEFAULT_SCHEDULER_INCLUDED_

#include "nbl/builtin/hlsl/scan/parameters_struct.hlsl"

#ifdef __cplusplus
#define uint uint32_t
#endif

namespace nbl
{
namespace hlsl
{
namespace scan
{	
	struct DefaultSchedulerParameters_t
	{
		uint finishedFlagOffset[NBL_BUILTIN_MAX_SCAN_LEVELS-1];
		uint cumulativeWorkgroupCount[NBL_BUILTIN_MAX_SCAN_LEVELS];

	};
}
}
}

#ifdef __cplusplus
#undef uint
#else

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
	 *							synchronizing between workgroups with while-loop spinning.
	 */
	void computeParameters(in uint elementCount, out Parameters_t _scanParams, out DefaultSchedulerParameters_t _schedulerParams)
	{
#define WorkgroupCount(Level) (_scanParams.lastElement[Level+1]+1u)
		_scanParams.lastElement[0] = elementCount-1u;
		_scanParams.topLevel = firstbithigh(_scanParams.lastElement[0])/_NBL_HLSL_WORKGROUP_SIZE_LOG2_;
		// REVIEW: _NBL_HLSL_WORKGROUP_SIZE_LOG2_ is defined in files that include THIS file. Why not query the API for workgroup size at runtime?
		
		for (uint i=0; i<NBL_BUILTIN_MAX_SCAN_LEVELS/2;)
		{
			const uint next = i+1;
			_scanParams.lastElement[next] = _scanParams.lastElement[i]>>_NBL_HLSL_WORKGROUP_SIZE_LOG2_;
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
#if NBL_BUILTIN_MAX_SCAN_LEVELS>7
#error "Switch needs more cases"
#endif
		}
#undef WorkgroupCount
	}
	
	/**
	 * treeLevel - the current level in the Blelloch Scan
	 * localWorkgroupIndex - the workgroup index the current invocation is a part of in the specific virtual dispatch. 
	 * For example, if we have dispatched 10 workgroups and we the virtual workgroup number is 35, then the localWorkgroupIndex should be 5.
	 */
	template<class ScratchAccessor>
	bool getWork(in DefaultSchedulerParameters_t params, in uint topLevel, out uint treeLevel, out uint localWorkgroupIndex)
	{
		ScratchAccessor sharedScratch;
		if(gl_LocalInvocationIndex == 0u) 
		{
			uint64_t original;
			InterlockedAdd(scanScratch.workgroupsStarted, 1u, original); // REVIEW: Refactor InterlockedAdd with GLSL terminology? // TODO (PentaKon): Refactor this when the ScanScratch descriptor set is declared
			sharedScratch.set(gl_LocalInvocationIndex, original);
		}
		else if (gl_LocalInvocationIndex == 1u) 
		{
			sharedScratch.set(gl_LocalInvocationIndex, 0u);
		}
		GroupMemoryBarrierWithGroupSync(); // REVIEW: refactor this somewhere with GLSL terminology?
		
		const uint globalWorkgroupIndex; // does every thread need to know?
		sharedScratch.get(0u, globalWorkgroupIndex);
		const uint lastLevel = topLevel<<1u;
		if (gl_LocalInvocationIndex<=lastLevel && globalWorkgroupIndex>=params.cumulativeWorkgroupCount[gl_LocalInvocationIndex]) 
		{
			InterlockedAdd(sharedScratch.get(1u, ?), 1u); // REVIEW: The way scratchaccessoradaptor is implemented (e.g. under subgroup/arithmetic_portability) doesn't allow for atomic ops on the scratch buffer. Should we ask for another implementation that overrides the [] operator ?
		}
		GroupMemoryBarrierWithGroupSync(); // TODO (PentaKon): Possibly refactor?
		
		sharedScratch.get(1u, treeLevel);
		if(treeLevel>lastLevel)
			return true;
		
		localWorkgroupIndex = globalWorkgroupIndex;
		const bool dependentLevel = treeLevel != 0u;
		if(dependentLevel) 
		{
			const uint prevLevel = treeLevel - 1u;
			localWorkgroupIndex -= params.cumulativeWorkgroupCount[prevLevel];
			if(gl_LocalInvocationIndex == 0u) 
			{
				uint dependentsCount = 1u;
				if(treeLevel <= topLevel) 
				{
					dependentsCount = _NBL_HLSL_WORKGROUP_SIZE_; // REVIEW: Defined in the files that include this file?
					const bool lastWorkgroup = (globalWorkgroupIndex+1u)==params.cumulativeWorkgroupCount[treeLevel];
					if (lastWorkgroup) 
					{
						const Parameters_t scanParams = getParameters(); // TODO (PentaKon): Undeclared as of now, this should return the Parameters_t from the push constants of (in)direct shader
						dependentsCount = scanParams.lastElement[treeLevel]+1u;
						if (treeLevel<topLevel) 
						{
							dependentsCount -= scanParams.lastElement[treeLevel+1u]*_NBL_HLSL_WORKGROUP_SIZE_;
						}
					}
				}
				uint dependentsFinishedFlagOffset = localWorkgroupIndex;
				if (treeLevel>topLevel) // !(prevLevel<topLevel) TODO: merge with `else` above?
					dependentsFinishedFlagOffset /= _NBL_HLSL_WORKGROUP_SIZE_;
				dependentsFinishedFlagOffset += params.finishedFlagOffset[prevLevel];
				while (scanScratch.data[dependentsFinishedFlagOffset]!=dependentsCount) // TODO (PentaKon): Refactor this when the ScanScratch descriptor set is declared
					GroupMemoryBarrierWithGroupSync(); // TODO (PentaKon): Possibly refactor?
			}
		}
		GroupMemoryBarrierWithGroupSync(); // TODO (PentaKon): Possibly refactor?
		return false;
	}
	
	void markComplete(in DefaultSchedulerParameters_t params, in uint topLevel, in uint treeLevel, in uint localWorkgroupIndex)
	{
		GroupMemoryBarrierWithGroupSync(); // must complete writing the data before flags itself as complete  // TODO (PentaKon): Possibly refactor?
		if (gl_LocalInvocationIndex==0u)
		{
			uint finishedFlagOffset = params.finishedFlagOffset[treeLevel];
			if (treeLevel<topLevel)
			{
				finishedFlagOffset += localWorkgroupIndex/_NBL_HLSL_WORKGROUP_SIZE_;
				InterlockedAdd(scanScratch.data[finishedFlagOffset],1u);
			}
			else if (treeLevel!=(topLevel<<1u))
			{
				finishedFlagOffset += localWorkgroupIndex;
				scanScratch.data[finishedFlagOffset] = 1u; // TODO (PentaKon): Refactor this when the ScanScratch descriptor set is declared
			}
		}
	}
}
}
}
}
#endif

#endif