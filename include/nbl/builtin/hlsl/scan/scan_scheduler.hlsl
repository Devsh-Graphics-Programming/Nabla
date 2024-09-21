// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_HLSL_SCAN_SCHEDULER_INCLUDED_
#define _NBL_HLSL_SCAN_SCHEDULER_INCLUDED_

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

template <uint16_t WorkgroupSize> struct ScanScheduler
{

    static ScanScheduler create(NBL_CONST_REF_ARG(Parameters_t) params, NBL_CONST_REF_ARG(DefaultSchedulerParameters_t) schedParams)
    {
        ScanScheduler<WorkgroupSize> scheduler;
        scheduler.params = params;
        scheduler.schedParams = schedParams;
        return scheduler;
    }

    template <class Accessor> bool getWork(NBL_REF_ARG(Accessor) accessor) // rename to hasWork()
    {

        // This part ensures that each workgroup that was not chosen to run the next level
        // will exit once all the work for the current level is finished
        glsl::memoryBarrierBuffer();
        if (scanScratchBuf[0u].workgroupsStarted[level] >= totalWorkgroupsPerLevel(level))
            return false;

        if (workgroup::SubgroupContiguousIndex() == 0u)
        {
            uberWorkgroupIndex = glsl::atomicAdd(scanScratchBuf[0u].workgroupsStarted[level], 1u);
        }
        uberWorkgroupIndex = workgroup::BroadcastFirst(uberWorkgroupIndex, accessor);
        // nothing really happens, we just check if there's a workgroup in the level to grab
        return uberWorkgroupIndex < totalWorkgroupsPerLevel(level);
    }

    uint32_t levelWorkgroupIndex(NBL_CONST_REF_ARG(uint32_t) level) { return uberWorkgroupIndex; }

    uint32_t totalWorkgroupsPerLevel(NBL_CONST_REF_ARG(uint32_t) level)
    {
        return level == 0u
            ? schedParams.cumulativeWorkgroupCount[level]
            : schedParams.cumulativeWorkgroupCount[level] - schedParams.cumulativeWorkgroupCount[level - 1u];
    }

    Parameters_t params;
    DefaultSchedulerParameters_t schedParams;
    uint32_t uberWorkgroupIndex; // rename to virtualWorkgroupIndex
    uint32_t level;              // 32 bit to stop warnings from level = workgroup::BroadcastFirst
};
}
}
}
#endif