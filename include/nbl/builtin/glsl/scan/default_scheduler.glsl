#ifndef _NBL_GLSL_SCAN_DEFAULT_SCHEDULER_INCLUDED_
#define _NBL_GLSL_SCAN_DEFAULT_SCHEDULER_INCLUDED_

#include <nbl/builtin/glsl/scan/parameters_struct.glsl>

#ifdef __cplusplus
#define uint uint32_t
#endif
struct nbl_glsl_scan_DefaultSchedulerParameters_t
{
	uint finishedFlagOffset[NBL_BUILTIN_MAX_SCAN_LEVELS-1];
	uint cumulativeWorkgroupCount[NBL_BUILTIN_MAX_SCAN_LEVELS];
};
#ifdef __cplusplus
#undef uint
#else

void nbl_glsl_scan_scheduler_computeParameters(in uint elementCount, out nbl_glsl_scan_Parameters_t scanParams, out nbl_glsl_scan_DefaultSchedulerParameters_t schedulerParams)
{
	scanParams.lastElement[0] = elementCount-1u;
	scanParams.lastElement[1] = scanParams.lastElement[0]>>_NBL_GLSL_WORKGROUP_SIZE_LOG2_;
	schedulerParams.finishedFlagOffset[0] = 0u;
	schedulerParams.cumulativeWorkgroupCount[0] = scanParams.lastElement[1]+1u;
	for (int i=1; i<NBL_BUILTIN_MAX_SCAN_LEVELS/2; )
	{
		const int next = i+1;
		scanParams.lastElement[next] = scanParams.lastElement[i]>>_NBL_GLSL_WORKGROUP_SIZE_LOG2_;
		const uint workgroupCount = scanParams.lastElement[next]+1u;
		schedulerParams.finishedFlagOffset[i] = schedulerParams.finishedFlagOffset[i-1]+workgroupCount;
		schedulerParams.cumulativeWorkgroupCount[i] = schedulerParams.cumulativeWorkgroupCount[i-1]+workgroupCount;
		i = next;
	}
	scanParams.topLevel = findMSB(scanParams.lastElement[0])/_NBL_GLSL_WORKGROUP_SIZE_LOG2_;
	{
		uint inoff = scanParams.topLevel;
		for (int i=0; i<NBL_BUILTIN_MAX_SCAN_LEVELS/2; i++)
		{
			const uint lastoff = inoff;
			inoff++;
			if (i!=(NBL_BUILTIN_MAX_SCAN_LEVELS/2-1))
				schedulerParams.finishedFlagOffset[inoff] = schedulerParams.finishedFlagOffset[lastoff];
			schedulerParams.cumulativeWorkgroupCount[inoff] = schedulerParams.cumulativeWorkgroupCount[lastoff];
			if (i<scanParams.topLevel)
			{
				schedulerParams.finishedFlagOffset[inoff] += scanParams.lastElement[scanParams.topLevel-i+1]+1u;
				schedulerParams.cumulativeWorkgroupCount[inoff] += scanParams.lastElement[scanParams.topLevel-i]+1u;
			}
		}
	}
	scanParams.temporaryStorageOffset[0] = scanParams.topLevel;
	if (scanParams.topLevel>1)
		scanParams.temporaryStorageOffset[0] = schedulerParams.finishedFlagOffset[(scanParams.topLevel<<1u)-1u]+scanParams.lastElement[2u]+1;
	for (int i=0; i<(NBL_BUILTIN_MAX_SCAN_LEVELS/2-1);)
	{
		const int next = i+1;
		scanParams.temporaryStorageOffset[next] = scanParams.temporaryStorageOffset[i]+scanParams.lastElement[next]+1u;
		i = next;
	}
}

bool nbl_glsl_scan_scheduler_getWork(in nbl_glsl_scan_DefaultSchedulerParameters_t params, in uint topLevel, out uint treeLevel, out uint localWorkgroupIndex)
{
	if (gl_LocalInvocationIndex==0u)
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = atomicAdd(scanScratch.workgroupsStarted,1u);
	else if (gl_LocalInvocationIndex==1u)
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = 0u;
	barrier();

	const uint globalWorkgroupIndex = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[0u]; // does every thread need to know?
	const uint lastLevel = topLevel<<1u;
	if (gl_LocalInvocationIndex<=lastLevel && globalWorkgroupIndex>=params.cumulativeWorkgroupCount[gl_LocalInvocationIndex])
		atomicAdd(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[1u],1u);
	barrier();

	treeLevel = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[1u];
	if (treeLevel>lastLevel)
		return true;
	
	localWorkgroupIndex = globalWorkgroupIndex;
	const bool dependantLevel = treeLevel!=0u;
	if (dependantLevel)
	{
		const uint prevLevel = treeLevel-1u;
		localWorkgroupIndex -= params.cumulativeWorkgroupCount[prevLevel];
		if (gl_LocalInvocationIndex==0u)
		{
			uint dependentsCount = 1u;
			if (treeLevel<=topLevel)
			{
				dependentsCount = _NBL_GLSL_WORKGROUP_SIZE_;
				const bool lastWorkgroup = (globalWorkgroupIndex+1u)==params.cumulativeWorkgroupCount[treeLevel];
				if (lastWorkgroup)
				{
					const nbl_glsl_scan_Parameters_t scanParams = nbl_glsl_scan_getParameters();
					dependentsCount = scanParams.lastElement[treeLevel]+1u;
					if (treeLevel<topLevel)
						dependentsCount -= scanParams.lastElement[treeLevel+1u]*_NBL_GLSL_WORKGROUP_SIZE_;
				}
			}

			uint dependentsFinishedFlagOffset = localWorkgroupIndex;
			if (treeLevel>topLevel) // !(prevLevel<topLevel) TODO: merge with `else` above?
				dependentsFinishedFlagOffset /= _NBL_GLSL_WORKGROUP_SIZE_;
			dependentsFinishedFlagOffset += params.finishedFlagOffset[prevLevel];
			while (scanScratch.data[dependentsFinishedFlagOffset]!=dependentsCount)
				memoryBarrierBuffer();
		}
	}
	barrier();
	return false;
}

void nbl_glsl_scan_scheduler_markComplete(in nbl_glsl_scan_DefaultSchedulerParameters_t params, in uint topLevel, in uint treeLevel, in uint localWorkgroupIndex)
{
	if (gl_LocalInvocationIndex==0u)
	{
		uint finishedFlagOffset = params.finishedFlagOffset[treeLevel];
		if (treeLevel<topLevel)
		{
			finishedFlagOffset += localWorkgroupIndex/_NBL_GLSL_WORKGROUP_SIZE_;
			atomicAdd(scanScratch.data[finishedFlagOffset],1u);
		}
		else if (treeLevel!=(topLevel<<1u))
		{
			finishedFlagOffset += localWorkgroupIndex;
			scanScratch.data[finishedFlagOffset] = 1u;
		}
	}
}
#endif

#endif