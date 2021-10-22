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

void nbl_glsl_scan_scheduler_computeParameters(in uint elementCount, out nbl_glsl_scan_Parameters_t _scanParams, out nbl_glsl_scan_DefaultSchedulerParameters_t _schedulerParams)
{
	_scanParams.lastElement[0] = elementCount-1u;
	_scanParams.topLevel = findMSB(_scanParams.lastElement[0])/_NBL_GLSL_WORKGROUP_SIZE_LOG2_;
	for (int i=0; i<NBL_BUILTIN_MAX_SCAN_LEVELS/2;)
	{
		const int next = i+1;
		_scanParams.lastElement[next] = _scanParams.lastElement[i]>>_NBL_GLSL_WORKGROUP_SIZE_LOG2_;
		i = next;
	}
#define WorkgroupCount(Level) (_scanParams.lastElement[Level+1]+1u)
	// comments have a worked example by hand
	_schedulerParams.cumulativeWorkgroupCount[0] = WorkgroupCount(0); // 102
	_schedulerParams.finishedFlagOffset[0] = 0u; // 0
	switch (_scanParams.topLevel)
	{
		case 1u:
			_schedulerParams.cumulativeWorkgroupCount[1] = _schedulerParams.cumulativeWorkgroupCount[0]+1u; // 103
			_schedulerParams.cumulativeWorkgroupCount[2] = _schedulerParams.cumulativeWorkgroupCount[1]+WorkgroupCount(0); // 205
			// climb up
			_schedulerParams.finishedFlagOffset[1] = 1u;
			//
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
			//
			_scanParams.temporaryStorageOffset[0] = _schedulerParams.finishedFlagOffset[3]+WorkgroupCount(1);
			_scanParams.temporaryStorageOffset[1] = _scanParams.temporaryStorageOffset[0]+WorkgroupCount(1);
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
			//
			_scanParams.temporaryStorageOffset[0] = _schedulerParams.finishedFlagOffset[5]+WorkgroupCount(1);
			_scanParams.temporaryStorageOffset[1] = _scanParams.temporaryStorageOffset[0]+WorkgroupCount(1);
			_scanParams.temporaryStorageOffset[2] = _scanParams.temporaryStorageOffset[1]+WorkgroupCount(2);
			break;
		default:
			break;
#if NBL_BUILTIN_MAX_SCAN_LEVELS>7
#error "Switch needs more cases"
#endif
	}
#undef WorkgroupCount
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
	memoryBarrierBuffer(); // ensure we read the previous workgroups data AFTER we read the finished flag
	return false;
}

void nbl_glsl_scan_scheduler_markComplete(in nbl_glsl_scan_DefaultSchedulerParameters_t params, in uint topLevel, in uint treeLevel, in uint localWorkgroupIndex)
{
	memoryBarrierBuffer(); // must complete writing the data before flags itself as complete
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