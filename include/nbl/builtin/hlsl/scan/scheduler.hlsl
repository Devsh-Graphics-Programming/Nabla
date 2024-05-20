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
  bool getWork(NBL_REF_ARG(Accessor) accessor)
  {
    if (workgroup::SubgroupContiguousIndex()==0u)
    {
      uberWorkgroupIndex = glsl::atomicAdd(scanScratchBuf[0u].workgroupsStarted,1u);
    }
    uberWorkgroupIndex = workgroup::BroadcastFirst(uberWorkgroupIndex, accessor);
    // nothing really happens, we just check if there's a workgroup in the level to grab
    return uberWorkgroupIndex <= schedParams.cumulativeWorkgroupCount[level] - 1u;
  }

  template<class Accessor>
  void markDone(NBL_REF_ARG(Accessor) accessor)
  {
    if (level==(params.topLevel)) // check if we reached the lastLevel
    {
      uberWorkgroupIndex = ~0u;
      return;
    }

    if (workgroup::SubgroupContiguousIndex()==0u)
    {
      // The uberWorkgroupIndex is always increasing, even after we switch levels, but for each new level the workgroupSetFinishedIndex must reset
      const uint32_t workgroupSetFinishedIndex = levelWorkgroupIndex(level) / WorkgroupSize;
      const uint32_t doneCount = glsl::atomicAdd(scanScratchBuf[0u].data[schedParams.workgroupFinishFlagsOffset[level]+workgroupSetFinishedIndex], 1u) + 1u;
      if ((uberWorkgroupIndex != schedParams.cumulativeWorkgroupCount[level] - 1u ? (WorkgroupSize-1u) : schedParams.lastWorkgroupSetCountForLevel[level])==doneCount)
      {
        level++;
      }
    }
    level = workgroup::BroadcastFirst(level, accessor);
  }

  uint32_t levelWorkgroupIndex(NBL_CONST_REF_ARG(uint32_t) level)
  {
      return level == 0u ? uberWorkgroupIndex : (uberWorkgroupIndex - schedParams.cumulativeWorkgroupCount[level-1u]);
  }

  Parameters_t params;
  DefaultSchedulerParameters_t schedParams;
  uint32_t uberWorkgroupIndex;
  uint16_t level;
};

}
}
}
#endif