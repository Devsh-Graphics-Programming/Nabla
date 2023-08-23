// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/workgroup/shared_ballot.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

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
template<class SharedAccessor, bool edgeBarriers = true>
void ballot(in bool value)
{
	SharedAccessor accessor;
	
	if(edgeBarriers)
		accessor.main.workgroupExecutionAndMemoryBarrier();
	
	uint initialize = gl_LocalInvocationIndex < uballotBitfieldCount;
	if(initialize) {
		accessor.main.set(gl_LocalInvocationIndex, 0u);
	}
	accessor.main.workgroupExecutionAndMemoryBarrier();
	if(value) {
		uint dummy;
		accessor.main.atomicOr(getDWORD(gl_LocalInvocationIndex), 1u<<(gl_LocalInvocationIndex&31u), dummy);
	}
	
	if(edgeBarriers)
		accessor.main.workgroupExecutionAndMemoryBarrier();
}

/**
 * Once we have assigned ballots in the shared array, we can 
 * extract any invocation's ballot value using this function.
 */
template<class SharedAccessor, bool edgeBarriers = true>
bool ballotBitExtract(in uint index)
{
	SharedAccessor accessor;
	
	if(edgeBarriers)
		accessor.main.workgroupExecutionAndMemoryBarrier();
	
	const bool retval = (accessor.main.get(getDWORD(index)) & (1u << (index & 31u))) != 0u;
	
	if(edgeBarriers)
		accessor.main.workgroupExecutionAndMemoryBarrier();
	
	return retval;
}

template<class SharedAccessor, bool edgeBarriers = true>
bool inverseBallot()
{
	return ballotBitExtract<SharedAccessor, edgeBarriers>(gl_LocalInvocationIndex);
}

/**
 * Gives us the sum of all ballots for the workgroup.
 *
 * Only the first few invocations are used for performing the sum 
 * since we only have `uballotBitfieldCount` amount of Uints that we need 
 * to add together.
 * 
 * We add them all in the shared array index after the last DWORD 
 * that is used for the ballots. For example, if we have 128 workgroup size,
 * then the array index in which we accumulate the sum is `4` since 
 * indexes 0..3 are used for ballots.
 */ 
template<class SharedAccessor>
uint ballotBitCount()
{
	SharedAccessor accessor;
	accessor.main.set(uballotBitfieldCount, 0u);
	accessor.main.workgroupExecutionAndMemoryBarrier();
	if(gl_LocalInvocationIndex < uballotBitfieldCount)
	{
		const uint localBallot = accessor.main.get(gl_LocalInvocationIndex);
		const uint localBallotBitCount = countbits(localBallot);
		uint dummy;
		accessor.main.atomicAdd(uballotBitfieldCount, localBallotBitCount, dummy);
	}
	accessor.main.workgroupExecutionAndMemoryBarrier();
	return accessor.main.get(uballotBitfieldCount);
}

}
}
}
#endif