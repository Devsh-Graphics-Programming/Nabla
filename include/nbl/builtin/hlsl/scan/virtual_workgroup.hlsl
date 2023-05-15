#ifndef _NBL_HLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_
#define _NBL_HLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_

// TODO (PentaKon): Decide if these are needed once we have a clearer picture of the refactor
#include "nbl/builtin/hlsl/limits/numeric.hlsl"
#include "nbl/builtin/hlsl/math/typeless_arithmetic.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl" // This is where all the nbl_glsl_workgroupOPs are defined
#include "nbl/builtin/hlsl/scan/declarations.hlsl"

#include "nbl/builtin/hlsl/binops.hlsl"

const uint gl_LocalInvocationIndex: SV_GroupIndex;

#if 0
namespace nbl
{
namespace hlsl
{
namespace scan
{
	template<class Binop, class Storage_t>
	void virtualWorkgroup(in uint treeLevel, in uint localWorkgroupIndex)
	{
		Binop binop;
		const Parameters_t params = getParameters();
		const uint levelInvocationIndex = localWorkgroupIndex * _NBL_HLSL_WORKGROUP_SIZE_ + gl_LocalInvocationIndex;
		const bool lastInvocationInGroup = gl_LocalInvocationIndex == (_NBL_HLSL_WORKGROUP_SIZE_ - 1);

		const uint lastLevel = params.topLevel << 1u;
		const uint pseudoLevel = levelInvocationIndex <= params.lastElement[pseudoLevel];

		const bool inRange = levelInvocationIndex <= params.lastElement[pseudoLevel];

		Storage_t data = binop.identity();
		if(inRange)
		{
			getData(data, levelInvocationIndex, localWorkgroupIndex, treeLevel, pseudoLevel);
		}

		if(treeLevel < params.topLevel) 
		{
			#error "Must also define some scratch accessor when calling operation()"
			data = workgroup::reduction<binop>()(data);
		}
		// REVIEW: missing _TYPE_ check and extra case here
		else if (treeLevel != params.topLevel)
		{
			data = workgroup::inclusive_scan<binop>()(data);
		}
		else
		{
			data = workgroup::exclusive_scan<binop>()(data);
		}
		setData(data, levelInvocationIndex, localWorkgroupIndex, treeLevel, pseudoLevel, inRange);
	}
}
}
}

#ifndef _NBL_HLSL_SCAN_MAIN_DEFINED_ // TODO REVIEW: Are these needed, can this logic be refactored?
#include "nbl/builtin/hlsl/scan/default_scheduler.hlsl"
namespace nbl
{
namespace hlsl
{
namespace scan
{
	DefaultSchedulerParameters_t getSchedulerParameters(); // this is defined in the final shader that assembles all the SCAN operation components
	void main()
	{
		const DefaultSchedulerParameters_t schedulerParams = getSchedulerParameters();
		const uint topLevel = getParameters().topLevel;
		// persistent workgroups
		while (true)
		{
			uint treeLevel,localWorkgroupIndex;
			if (scheduler::getWork(schedulerParams,topLevel,treeLevel,localWorkgroupIndex))
			{
				return;
			}

			virtualWorkgroup(treeLevel,localWorkgroupIndex);

			scheduler::markComplete(schedulerParams,topLevel,treeLevel,localWorkgroupIndex);
		}
	}
}
}
}
#endif

#define _NBL_HLSL_SCAN_MAIN_DEFINED_
#endif

#endif