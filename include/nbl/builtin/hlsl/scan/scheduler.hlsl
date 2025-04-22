// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_SCHEDULER_INCLUDED_
#define _NBL_HLSL_SCHEDULER_INCLUDED_

#include "nbl/builtin/hlsl/scan/declarations.hlsl"
#include "nbl/builtin/hlsl/scan/descriptors.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"

namespace nbl
{
namespace hlsl
{
namespace scan
{

template<uint16_t WorkgroupSize>
struct Scheduler
{
    static Scheduler create(NBL_CONST_REF_ARG(Parameters_t) params, NBL_CONST_REF_ARG(DefaultSchedulerParameters_t) schedParams)
    {
        Scheduler<WorkgroupSize> scheduler;
        scheduler.params = params;
        scheduler.schedParams = schedParams;
        return scheduler;
    }
    
    template<class Accessor>
    bool getWork(NBL_REF_ARG(Accessor) accessor)  // rename to hasWork()
    {
        
        // This part ensures that each workgroup that was not chosen to run the next level 
        // will exit once all the work for the current level is finished
        glsl::memoryBarrierBuffer();
        if(scanScratchBuf[0u].workgroupsStarted[level] >= totalWorkgroupsPerLevel(level))
            return false;
    
        if (workgroup::SubgroupContiguousIndex()==0u)
        {
            uberWorkgroupIndex = glsl::atomicAdd(scanScratchBuf[0u].workgroupsStarted[level],1u);
        }
        uberWorkgroupIndex = workgroup::BroadcastFirst(uberWorkgroupIndex, accessor);
        // nothing really happens, we just check if there's a workgroup in the level to grab
        return uberWorkgroupIndex < totalWorkgroupsPerLevel(level);
    }

    template<class Accessor>
    bool markDone(NBL_REF_ARG(Accessor) accessor)
    {
        if (level==(params.topLevel)) // check if we reached the lastLevel
        {
            // not even sure this is needed
            uberWorkgroupIndex = ~0u;
            return true;
        }

        uint32_t prevLevel = level;
        if (workgroup::SubgroupContiguousIndex()==0u)
        {
            // The uberWorkgroupIndex is always increasing, even after we switch levels, but for each new level the workgroupSetFinishedIndex must reset
            const uint32_t workgroupSetFinishedIndex = levelWorkgroupIndex(level) / WorkgroupSize;
            const uint32_t lastWorkgroupSetIndexForLevel = totalWorkgroupsPerLevel(level) / WorkgroupSize;
            const uint32_t doneCount = glsl::atomicAdd(scanScratchBuf[0u].data[schedParams.workgroupFinishFlagsOffset[level]+workgroupSetFinishedIndex], 1u);
            //if ((uberWorkgroupIndex != schedParams.cumulativeWorkgroupCount[level] - 1u ? (WorkgroupSize-1u) : schedParams.lastWorkgroupSetCountForLevel[level])==doneCount)
            if (((uberWorkgroupIndex/WorkgroupSize) < lastWorkgroupSetIndexForLevel  ? (WorkgroupSize-1u) : schedParams.lastWorkgroupSetCountForLevel[level])==doneCount)
            {
                level++;
            }
        }
        level = workgroup::BroadcastFirst(level, accessor);
        return level == 0 ? false : level == prevLevel; // on level 0 never exit early but on higher levels each workgroup is allowed one operation and exits except if promoted to next level
    }

    uint32_t levelWorkgroupIndex(NBL_CONST_REF_ARG(uint32_t) level)
    {
        return uberWorkgroupIndex;
    }

    uint32_t totalWorkgroupsPerLevel(NBL_CONST_REF_ARG(uint32_t) level)
    {
        return level == 0 
            ? schedParams.cumulativeWorkgroupCount[level] 
            : schedParams.cumulativeWorkgroupCount[level] - schedParams.cumulativeWorkgroupCount[level-1u];
    }
    
    void resetWorkgroupsStartedBuffer()
    {
        // could do scanScratchBuf[0u].workgroupsStarted[SubgroupContiguousIndex()] = 0u but don't know how many invocations are live during this call
        if(workgroup::SubgroupContiguousIndex() == 0u)
        {
            for(uint32_t i = 0; i < params.topLevel; i++)
            {
                scanScratchBuf[0u].workgroupsStarted[i] = 0u;
            }
        }
    }

  Parameters_t params;
  DefaultSchedulerParameters_t schedParams;
  uint32_t uberWorkgroupIndex; // rename to virtualWorkgroupIndex
  uint32_t level; // 32 bit to stop warnings from level = workgroup::BroadcastFirst
};

}
}
}
#endif