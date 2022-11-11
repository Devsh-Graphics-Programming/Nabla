// Copyright (C) 2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SCANNING_APPEND_INCLUDED_
#define _NBL_BUILTIN_HLSL_SCANNING_APPEND_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace scanning_append
{

struct result_t
{
    static result_t invalid()
    {
        result_t retval;
        retval.exclusivePrefixSum = retval.outputIndex = ~0u;
        return retval;
    }

    uint outputIndex;
    uint exclusivePrefixSum;
};


// Elements with value 0 do not get appended
// Note: If NBL_GLSL_EXT_shader_atomic_int64 is not present, then the call to these functions needs to be subgroup uniform
template<class AtomicCounterAccessor>
result_t non_negative(inout AtomicCounterAccessor accessor, in uint value)
{
  const bool willAppend = bool(value);

  result_t retval;
#ifdef NBL_GLSL_EXT_shader_atomic_int64
  uint64_t add = value;
  if (willAppend)
     add |= 0x100000000ull;
  const uint64_t count_reduction = accessor.fetchIncr(add);
  retval.outputIndex = uint(count_reduction>>32);
  retval.exclusivePrefixSum = uint(count_reduction);
#else
  #error "Untested Path, won't touch this until we actually need to ship something on Vulkan mobile or GL!"
  uint localIndex = subgroup::ballotExclusiveBitCount(subgroup::ballot(willAppend));
  uint partialPrefix = subgroup::exclusiveAdd(value);

  uint subgroupIndex,subgroupPrefix;
  // elect last invocation
  const uint lastSubgroupInvocationID = subgroup::Size-1u;
  if (subgroup::InvocationID==lastSubgroupInvocationID)
  {
    // crude mutex, reuse MSB bit
    const uint lockBit = 0x80000000u;
    // first subgroup to set the bit to 1 (old value 0) proceeds with the lock
    while (accessor.fetchOrCount(lockBit)) {}
    // now MSB is always 1
    subgroupPrefix = accessor.fetchIncrSum(partialPrefix+value);
    // set the MSB to 0 (unlock) while adding, by making sure MSB overflows
    uint subgroupCount = localIndex;
    if (willAppend)
        subgroupCount++;
    subgroupIndex = accessor.fetchIncrCount(lockBit|subgroupCount);
  }
  retval.outputIndex = subgroup::broadcast(subgroupIndex,lastSubgroupInvocationID)+localIndex;
  retval.exclusivePrefixSum = subgroup::broadcast(subgroupPrefix,lastSubgroupInvocationID)+partialPrefix;
#endif
  return retval;
}

// optimized version which tries to omit the atomicAdd and locks if it can, in return it may return garbage/invalid value when invocation's `value==0`
template<class AtomicCounterAccessor>
result_t positive(inout AtomicCounterAccessor accessor, in uint value)
{
  const bool willAppend = bool(value);
#ifdef NBL_GLSL_EXT_shader_atomic_int64
  if (willAppend)
#else
  if (WaveActiveAnyTrue(willAppend))
#endif
    return non_negative<AtomicCounterAccessor>(accessor,value);

  return result_t::invalid();
}

}
}
}

#endif
