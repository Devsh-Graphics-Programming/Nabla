#ifndef _NBL_HLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_
#define _NBL_HLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_

// TODO (PentaKon): Decide if these are needed once we have a clearer picture of the refactor
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl" // This is where all the nbl_glsl_workgroupOPs are defined
#include "nbl/builtin/hlsl/scan/declarations.hlsl"
#include "nbl/builtin/hlsl/scan/default_scheduler.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{
	template<class Binop, typename Storage_t, bool isExclusive, uint16_t ItemCount, class Accessor, class device_capabilities=void>
	void virtualWorkgroup(NBL_CONST_REF_ARG(uint32_t) treeLevel, NBL_CONST_REF_ARG(uint32_t) localWorkgroupIndex, NBL_REF_ARG(Accessor) accessor)
	{
		const Parameters_t params = getParameters();
		const uint32_t levelInvocationIndex = localWorkgroupIndex * glsl::gl_WorkGroupSize().x + workgroup::SubgroupContiguousIndex();
		const bool lastInvocationInGroup = workgroup::SubgroupContiguousIndex() == (glsl::gl_WorkGroupSize().x - 1);

		const uint32_t lastLevel = params.topLevel << 1u;
		const uint32_t pseudoLevel = treeLevel>params.topLevel ? (lastLevel-treeLevel):treeLevel;
		const bool inRange = levelInvocationIndex <= params.lastElement[pseudoLevel];

        // REVIEW: Right now in order to support REDUCE operation we need to set the max treeLevel == topLevel
        // so that it exits after reaching the top?

		Storage_t data = Binop::identity; // REVIEW: replace Storage_t with Binop::type_t?
		if(inRange)
		{
			getData<Storage_t, isExclusive>(data, levelInvocationIndex, localWorkgroupIndex, treeLevel, pseudoLevel);
		}

		if(treeLevel < params.topLevel) 
		{
            data = workgroup::reduction<Binop,ItemCount,device_capabilities>::template __call<Accessor>(data,accessor);
		}
        else if (!isExclusive && params.topLevel == 0u)
        {
            data = workgroup::inclusive_scan<Binop,ItemCount,device_capabilities>::template __call<Accessor>(data,accessor);
        }
		else if (treeLevel != params.topLevel)
		{
			data = workgroup::inclusive_scan<Binop,ItemCount,device_capabilities>::template __call<Accessor>(data,accessor);
		}
		else
		{
			data = workgroup::exclusive_scan<Binop,ItemCount,device_capabilities>::template __call<Accessor>(data,accessor);
		}
		setData(data, levelInvocationIndex, localWorkgroupIndex, treeLevel, pseudoLevel, inRange);
	}

	DefaultSchedulerParameters_t getSchedulerParameters(); // this is defined in the final shader that assembles all the SCAN operation components
	template<class Binop, typename Storage_t, bool isExclusive, uint16_t ItemCount, class Accessor>
    void main(NBL_REF_ARG(Accessor) accessor)
	{
		const DefaultSchedulerParameters_t schedulerParams = getSchedulerParameters();
		const uint32_t topLevel = getParameters().topLevel;
		// persistent workgroups
		while (true)
		{
            // REVIEW: Need to create accessor here.
            // REVIEW: Regarding ItemsPerWG this must probably be calculated after each getWork call?
			uint32_t treeLevel,localWorkgroupIndex;
			if (scheduler::getWork<Accessor>(schedulerParams, topLevel, treeLevel, localWorkgroupIndex, accessor))
			{
				return;
			}

			virtualWorkgroup<Binop, Storage_t, isExclusive, ItemCount, Accessor>(treeLevel, localWorkgroupIndex, accessor);

			scheduler::markComplete(schedulerParams,topLevel,treeLevel,localWorkgroupIndex);
		}
	}
}
}
}

#endif