// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{
namespace impl
{
uint getDWORD(uint invocation)
{
    return invocation >> 5;
}
}
// uballotBitfieldCount essentially means 'how many DWORDs are needed to store ballots in bitfields, for each invocation of the workgroup'
// can't use getDWORD because we want the static const to be treated as 'constexpr'
static const uint uballotBitfieldCount = (_NBL_HLSL_WORKGROUP_SIZE_+31) >> 5; // in case WGSZ is not a multiple of 32 we might miscalculate the DWORDs after the right-shift by 5 which is why we add 31

/**
 * Simple ballot function.
 *
 * Each invocation provides a boolean value. Each value is represented by a 
 * single bit of a Uint. For example, if invocation index 5 supplies `value = true` 
 * then the Uint will be ...00100000
 * This way we can encode 32 invocations into a single Uint.
 *
 * All Uints are kept in contiguous accessor memory in a shared array.
 * The size of that array is based on the WORKGROUP SIZE. In this case we use uballotBitfieldCount.
 *
 * For each group of 32 invocations, a DWORD is assigned to the array (i.e. a 32-bit value, in this case Uint).
 * For example, for a workgroup size 128, 4 DWORDs are needed.
 * For each invocation index, we can find its respective DWORD index in the accessor array 
 * by calling the getDWORD function.
 */
template<class SharedAccessor>
void ballot(const bool value, NBL_REF_ARG(SharedAccessor) accessor)
{
    uint initialize = gl_LocalInvocationIndex < uballotBitfieldCount;
    if(initialize) {
        accessor.ballot.set(gl_LocalInvocationIndex, 0u);
    }
    accessor.ballot.workgroupExecutionAndMemoryBarrier();
    if(value) {
        uint dummy;
        accessor.ballot.atomicOr(impl::getDWORD(gl_LocalInvocationIndex), 1u<<(gl_LocalInvocationIndex&31u), dummy);
    }
}

template<class SharedAccessor>
bool ballotBitExtract(const uint index, NBL_REF_ARG(SharedAccessor) accessor)
{
    return (accessor.ballot.get(impl::getDWORD(index)) & (1u << (index & 31u))) != 0u;
}

/**
 * Once we have assigned ballots in the shared array, we can 
 * extract any invocation's ballot value using this function.
 */
template<class SharedAccessor>
bool inverseBallot(NBL_REF_ARG(SharedAccessor) accessor)
{
    return ballotBitExtract<SharedAccessor>(gl_LocalInvocationIndex, accessor);
}

}
}
}
#endif