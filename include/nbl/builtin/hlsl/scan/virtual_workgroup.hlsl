#ifndef _NBL_HLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_
#define _NBL_HLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_

// TODO (PentaKon): Decide if these are needed once we have a clearer picture of the refactor
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl" // This is where all the nbl_glsl_workgroupOPs are defined
#include "nbl/builtin/hlsl/scan/declarations.hlsl"
#include "nbl/builtin/hlsl/scan/scheduler.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{
    template<class Binop, typename Storage_t, bool isScan, bool isExclusive, uint16_t WorkgroupSize, class Accessor, class device_capabilities=void>
    void virtualWorkgroup(NBL_CONST_REF_ARG(uint32_t) treeLevel, NBL_CONST_REF_ARG(uint32_t) levelWorkgroupIndex, NBL_REF_ARG(Accessor) accessor)
    {
        const Parameters_t params = getParameters();
        const uint32_t levelInvocationIndex = levelWorkgroupIndex * glsl::gl_WorkGroupSize().x + workgroup::SubgroupContiguousIndex();
        const uint32_t lastLevel = params.topLevel << 1u;
        
        // pseudoLevel is the level index going up until toplevel then back down
        const uint32_t pseudoLevel = treeLevel>params.topLevel ? (lastLevel-treeLevel):treeLevel;
        const bool inRange = levelInvocationIndex <= params.lastElement[pseudoLevel];

        // REVIEW: Right now in order to support REDUCE operation we need to set the max treeLevel == topLevel
        // so that it exits after reaching the top?
        
        // Seems that even though it's a mem barrier is must NOT diverge!
        // This was called inside getData but due to inRange it can possibly diverge 
        // so it must be called outside! Not yet sure why a mem barrier must be uniformly called.
        glsl::memoryBarrierBuffer(); // scanScratchBuf can't be declared as coherent due to VMM(?)
        
        Storage_t data = Binop::identity; // REVIEW: replace Storage_t with Binop::type_t?
        if(inRange)
        {
            getData<Storage_t, isExclusive>(data, levelInvocationIndex, levelWorkgroupIndex, treeLevel, pseudoLevel);
        }

        bool doReduce = isScan ? treeLevel < params.topLevel : treeLevel <= params.topLevel;

        if(doReduce)
        {
            data = workgroup::reduction<Binop,WorkgroupSize,device_capabilities>::template __call<Accessor>(data,accessor);
        }
        else if (!isExclusive && params.topLevel == 0u)
        {
            data = workgroup::inclusive_scan<Binop,WorkgroupSize,device_capabilities>::template __call<Accessor>(data,accessor);
        }
        else if (treeLevel != params.topLevel)
        {
            data = workgroup::inclusive_scan<Binop,WorkgroupSize,device_capabilities>::template __call<Accessor>(data,accessor);
        }
        else
        {
            data = workgroup::exclusive_scan<Binop,WorkgroupSize,device_capabilities>::template __call<Accessor>(data,accessor);
        }
        setData<Storage_t, isScan>(data, levelInvocationIndex, levelWorkgroupIndex, treeLevel, pseudoLevel, inRange);
    }

    DefaultSchedulerParameters_t getSchedulerParameters(); // this is defined in the final shader that assembles all the SCAN operation components
    template<class Binop, typename Storage_t, bool isScan, bool isExclusive, uint16_t WorkgroupSize, class Accessor>
    void main(NBL_REF_ARG(Accessor) accessor)
    {
        const Parameters_t params = getParameters();
        const DefaultSchedulerParameters_t schedulerParams = getSchedulerParameters();
        Scheduler<WorkgroupSize> scheduler = Scheduler<WorkgroupSize>::create(params, schedulerParams);
        // persistent workgroups
        while (true)
        {
            if (!scheduler.getWork(accessor))
            {
                return;
            }

            virtualWorkgroup<Binop, Storage_t, isScan, isExclusive, WorkgroupSize, Accessor>(scheduler.level, scheduler.levelWorkgroupIndex(scheduler.level), accessor);
            accessor.workgroupExecutionAndMemoryBarrier();
            scheduler.markDone(accessor);
        }
    }
}
}
}

#endif