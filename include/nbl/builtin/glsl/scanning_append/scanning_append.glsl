// No header guards!!! In case you want to have multiple append buffers,
// you might want to `#include` this header multiple times with different suffices
#ifndef NBL_GLSL_SCANNING_APPEND_FUNCNAME_SUFFFIX
#define NBL_GLSL_SCANNING_APPEND_FUNCNAME_SUFFFIX
#endif

// LSB
#ifdef NBL_GLSL_EXT_shader_atomic_int64
#define nbl_glsl_scanning_append_counter_t uint64_t
#else
#define nbl_glsl_scanning_append_counter_t uvec2
#endif

struct nbl_glsl_scanning_append_result_t
{
   uint outIndex;
   uint exclusivePrefix;
};

// Elements with value 0 do not get appended
// Note: If NBL_GLSL_EXT_shader_atomic_int64 is not present, then the call to this function needs to be subgroup uniform
nbl_glsl_scanning_append_result_t NBL_GLSL_CONCATENATE(nbl_glsl_scanning_append,NBL_GLSL_SCANNING_APPEND_FUNCNAME_SUFFFIX)(in uint value)
{
#ifndef NBL_GLSL_SCANNING_APPEND_COUNTER_NAME
#error "Need to define NBL_GLSL_SCANNING_APPEND_COUNTER_NAME for the `nbl_glsl_scanning_append` function, cause GLSL is dumb and `buffer` cannot be passed around."
#endif

  const bool willAppend = bool(value);
#ifdef NBL_GLSL_EXT_shader_atomic_int64
  uint64_t add = value;
  if (willAppend)
     add |= 0x100000000ull;
  const uint64_t count_reduction = atomicAdd(NBL_GLSL_SCANNING_APPEND_COUNTER_NAME,add);
  return nbl_glsl_scanning_append_result_t{uint(count_reduction>>32),uint(count_reduction)};
#else
  #error "Untested Path, won't touch this until we actually need to ship something on Vulkan mobile or GL!"
  uint localIndex = nbl_glsl_subgroupBallotExclusiveBitCount(nbl_glsl_subgroupBallot(willAppend));
  uint partialPrefix = nbl_glsl_subgroupExclusiveAdd(value);

  const uint lastSubgroupInvocationID = nbl_glsl_SubgroupSize-1u;
  uint subgroupIndex,subgroupPrefix;
  if (nbl_glsl_SubgroupInvocationID==lastSubgroupInvocationID)
  {
    // crude mutex, reuse MSB bit
    const uint lockBit = 0x80000000u;
    // first subgroup to set the bit to 1 (old value 0) proceeds with the lock
    while (bool(atomicOr(NBL_GLSL_SCANNING_APPEND_COUNTER_NAME[1],lockBit))) {}
    // now MSB is always 1
    subgroupPrefix = atomicAdd(NBL_GLSL_SCANNING_APPEND_COUNTER_NAME[0],partialPrefix+value);
    // set the MSB to 0 while adding by making sure MSB overflows
    subgroupIndex = atomicAdd(NBL_GLSL_SCANNING_APPEND_COUNTER_NAME[1],localIndex+(willAppend ? (lockBit+1):lockBit));
  }
  return nbl_glsl_scanning_append_result_t{
    nbl_glsl_subgroupBroadcast(subgroupIndex,lastSubgroupInvocationID)+localIndex,
    nbl_glsl_subgroupBroadcast(subgroupPrefix,lastSubgroupInvocationID)+partialPrefix
  };
#endif
}

// optimized version which tries to omit the atomicAdd and locks if it can, in return it may return garbage/invalid value when invocation's `value==0`
nbl_glsl_scanning_append_result_t NBL_GLSL_CONCATENATE(nbl_glsl_scanning_append,NBL_GLSL_SCANNING_APPEND_FUNCNAME_SUFFFIX)(in uint value)
{
  const bool willAppend = bool(value);
#ifdef NBL_GLSL_EXT_shader_atomic_int64
  if (willAppend)
#else
  if (subgroupAny(willAppend))
#endif
    return NBL_GLSL_CONCATENATE(nbl_glsl_scanning_append,NBL_GLSL_SCANNING_APPEND_FUNCNAME_SUFFFIX)(value);

  return nbl_glsl_scanning_append_result_t{~0u,~0u};
}

#undef NBL_GLSL_SCANNING_APPEND_FUNCNAME_SUFFFIX
