#ifndef _NBL_HLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_
#define _NBL_HLSL_SCAN_VIRTUAL_WORKGROUP_INCLUDED_

// TODO (PentaKon): Decide if these are needed once we have a clearer picture of the refactor
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl" // This is where all the nbl_glsl_workgroupOPs are defined
#include "nbl/builtin/hlsl/scan/declarations.hlsl"
#include "nbl/builtin/hlsl/scan/scheduler.hlsl"
#include "nbl/builtin/hlsl/scan/scan_scheduler.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{   
    /**
     * Finds the index on a higher level in the reduce tree which the 
     * levelInvocationIndex needs to add to level 0
     */
    template<uint32_t WorkgroupSize>
    uint32_t2 findHigherLevelIndex(uint32_t level, uint32_t levelInvocationIndex)
    {
		const uint32_t WgSzLog2 = firstbithigh(WorkgroupSize);
		uint32_t rightIndexNonInclusive = (levelInvocationIndex >> (WgSzLog2 * level)); // formula for finding the index. If it underflows it means that there's nothing to do for the level
		uint32_t leftIndexInclusive = (rightIndexNonInclusive >> WgSzLog2) << WgSzLog2;
		uint32_t add = (levelInvocationIndex & (WgSzLog2 * level - 1)) > 0 ? 1u : 0u; // if the index mod WorkgroupSize*level > 0 then we must add 1u
		return uint32_t2(leftIndexInclusive, rightIndexNonInclusive); // return both the index and the potential underflow information to know we must do nothing
    }

    template<class Binop, typename Storage_t, bool isScan, bool isExclusive, uint16_t WorkgroupSize, class Accessor, class device_capabilities=void>
    void virtualWorkgroup(NBL_CONST_REF_ARG(uint32_t) treeLevel, NBL_CONST_REF_ARG(uint32_t) levelWorkgroupIndex, NBL_REF_ARG(Accessor) accessor)
    {
        const Parameters_t params = getParameters();
        const uint32_t levelInvocationIndex = levelWorkgroupIndex * glsl::gl_WorkGroupSize().x + workgroup::SubgroupContiguousIndex();
        
        const bool inRange = levelInvocationIndex <= params.lastElement[treeLevel]; // the lastElement array contains the lastElement's index hence the '<='

        // REVIEW: Right now in order to support REDUCE operation we need to set the max treeLevel == topLevel
        // so that it exits after reaching the top?
        
        // Seems that even though it's a mem barrier is must NOT diverge!
        // This was called inside getData but due to inRange it can possibly diverge 
        // so it must be called outside! Not yet sure why a mem barrier must be uniformly called.
        glsl::memoryBarrierBuffer(); // scanScratchBuf can't be declared as coherent due to VMM(?)
        
        Storage_t data = Binop::identity; // REVIEW: replace Storage_t with Binop::type_t?
        if(inRange)
        {
            getData<Storage_t, isExclusive>(data, levelInvocationIndex, levelWorkgroupIndex, treeLevel);
        }

        bool doReduce = !isScan;
        if(doReduce)
        {
            data = workgroup::reduction<Binop,WorkgroupSize,device_capabilities>::template __call<Accessor>(data,accessor);
        }
        else 
        {
            if (isExclusive)
            {
                data = workgroup::exclusive_scan<Binop,WorkgroupSize,device_capabilities>::template __call<Accessor>(data,accessor);
            }
            else
            {
                data = workgroup::inclusive_scan<Binop,WorkgroupSize,device_capabilities>::template __call<Accessor>(data,accessor);
            }
            
            uint32_t reverseLevel = params.topLevel - treeLevel;
            while(reverseLevel != ~0u)
            {
                Storage_t tempData = Binop::identity;
                uint32_t2 idxRange = findHigherLevelIndex<uint32_t(WorkgroupSize)>(reverseLevel--, levelInvocationIndex);
                if(workgroup::SubgroupContiguousIndex() < idxRange.y - idxRange.x)
                {
                    tempData = scanScratchBuf[0].data[params.temporaryStorageOffset[reverseLevel] + idxRange.x + workgroup::SubgroupContiguousIndex()];
                }

                // we could potentially do an inclusive scan of this part and cache it for other WGs
                //tempData = workgroup::inclusive_scan<Binop,WorkgroupSize,device_capabilities>::template __call<Accessor>(tempData,accessor);
                data += workgroup::reduction<Binop,WorkgroupSize,device_capabilities>::template __call<Accessor>(tempData,accessor);
            }
        }
        
        setData<Storage_t, isScan>(data, levelInvocationIndex, levelWorkgroupIndex, treeLevel, inRange);
    }

    DefaultSchedulerParameters_t getSchedulerParameters(); // this is defined in the final shader that assembles all the SCAN operation components
    template<class Binop, typename Storage_t, bool isScan, bool isExclusive, uint16_t WorkgroupSize, class Accessor>
    void main(NBL_REF_ARG(Accessor) accessor)
    {
        const Parameters_t params = getParameters();
        const DefaultSchedulerParameters_t schedulerParams = getSchedulerParameters();
        
        if(!isScan)
        {
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
                if(scheduler.markDone(accessor))
                {
                    return;
                }
            }
        }
        else
        {
            ScanScheduler<WorkgroupSize> scheduler = ScanScheduler<WorkgroupSize>::create(params, schedulerParams);
            while(scheduler.getWork(accessor))
            {
                virtualWorkgroup<Binop, Storage_t, isScan, isExclusive, WorkgroupSize, Accessor>(scheduler.level, scheduler.levelWorkgroupIndex(scheduler.level), accessor);
                accessor.workgroupExecutionAndMemoryBarrier();
            }
        }
    }
}
}
}

#endif