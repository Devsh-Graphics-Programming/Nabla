#ifndef _NBL_GLSL_SCAN_DEFAULT_SCHEDULER_INCLUDED_
#define _NBL_GLSL_SCAN_DEFAULT_SCHEDULER_INCLUDED_

#include <nbl/builtin/glsl/scan/parameters_struct.glsl>

#ifdef __cplusplus
#define uint uint32_t
#endif
struct nbl_glsl_scan_DefaultSchedulerParameters_t
{
	uint lastWorkgroupDependentCount[NBL_BUILTIN_MAX_SCAN_LEVELS/2];
	uint finishedFlagOffset[NBL_BUILTIN_MAX_SCAN_LEVELS-1];
	uint cumulativeWorkgroupCount[NBL_BUILTIN_MAX_SCAN_LEVELS];
};
#ifdef __cplusplus
#undef uint
#else
bool nbl_glsl_scan_scheduler_getWork(in nbl_glsl_scan_DefaultSchedulerParameters_t params, in uint topLevel, out uint treeLevel, out uint localWorkgroupIndex)
{
	if (gl_LocalInvocationIndex==0u)
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = atomicAdd(scanScratch.workgroupsStarted,1u);
	else if (gl_LocalInvocationIndex==1u)
		_NBL_GLSL_SCRATCH_SHARED_DEFINED_[gl_LocalInvocationIndex] = 0u;
	barrier();

	const uint globalWorkgroupIndex = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[0u]; // does every thread need to know?
	if (gl_LocalInvocationIndex<NBL_BUILTIN_MAX_SCAN_LEVELS && globalWorkgroupIndex>=params.cumulativeWorkgroupCount[gl_LocalInvocationIndex])
		atomicAdd(_NBL_GLSL_SCRATCH_SHARED_DEFINED_[1u],1u);
	barrier();

	treeLevel = _NBL_GLSL_SCRATCH_SHARED_DEFINED_[1u];
	if (treeLevel>(topLevel<<1u))
		return true;
	
	localWorkgroupIndex = globalWorkgroupIndex;
	const bool dependantLevel = treeLevel!=0u;
	if (dependantLevel)
	{
		const uint prevLevel = treeLevel-1u;
		localWorkgroupIndex -= params.cumulativeWorkgroupCount[prevLevel];
		if (gl_LocalInvocationIndex==0u)
		{
			uint dependentsCount;
			if (prevLevel<topLevel)
			{
				const bool lastWorkgroup = (globalWorkgroupIndex+1u)==params.cumulativeWorkgroupCount[treeLevel];
				dependentsCount = lastWorkgroup ? params.lastWorkgroupDependentCount[prevLevel]:_NBL_GLSL_WORKGROUP_SIZE_;
			}
			else
				dependentsCount = 1u;

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