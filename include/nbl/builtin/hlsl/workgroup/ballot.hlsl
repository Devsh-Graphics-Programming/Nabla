// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/atomics.hlsl"
#include "nbl/builtin/hlsl/workgroup/basic.hlsl"
#include "nbl/builtin/hlsl/subgroup/scratch.hlsl"
#include "nbl/builtin/hlsl/workgroup/shared_scan.hlsl"
#include "nbl/builtin/hlsl/subgroup/arithmetic_portability.hlsl"

#ifndef _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
#define _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
const uint gl_LocalInvocationIndex : SV_GroupIndex;
#endif

// REVIEW: They were defined in shared_ballot.glsl but only used in ballot.glsl so they were moved
#define getDWORD(IX) ((IX)>>5)
// bitfieldDWORDs essentially means 'how many DWORDs are needed to store ballots in bitfields, for each invocation of the workgroup'
#define bitfieldDWORDs getDWORD(_NBL_HLSL_WORKGROUP_SIZE_+31)

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
 * All Uints are kept in contiguous scratch memory in a shared array.
 * The size of that array is based on the WORKGROUP SIZE. In this case we use bitfieldDWORDs.
 *
 * For each group of 32 invocations, a DWORD is assigned to the array (i.e. a 32-bit value, in this case Uint).
 * For example, for a workgroup size 128, 4 DWORDs are needed.
 * For each invocation index, we can find its respective DWORD index in the scratch array 
 * by calling the getDWORD function.
 */
template<class ScratchAccessor, bool initialize, bool edgeBarriers> // REVIEW: Not sure if initialization should be done as a template or as a runtime check. Not sure if barriers as tpl or as new function 'ballotWithBarriers'
// initialize = gl_LocalInvocationIndex < bitfieldDWORDs
void ballot(in bool value)
{
	if(edgeBarriers)
		Barrier();
	
	ScratchAccessor scratch;
	if(initialize) {
		scratch.set(gl_LocalInvocationIndex, 0u);
	}
	Barrier();
	if(value) {
		uint temp;
		scratch.atomicOr(getDWORD(gl_LocalInvocationIndex), 1u<<(gl_LocalInvocationIndex&31u), temp);
	}
	
	if(edgeBarriers)
		Barrier();
}

/**
 * Once we have assigned ballots in the scratch array, we can 
 * extract any invocation's ballot value using this function.
 */
template<class ScratchAccessor, bool edgeBarriers>
bool ballotBitExtract(in uint index) // REVIEW: Should this just be incorporated inside inverseBallot() ?
{
	if(edgeBarriers)
		Barrier();
	
	ScratchAccessor scratch;
	const bool retval = (scratch.get(getDWORD(index)) & (1u << (index & 31u))) != 0u;
	
	if(edgeBarriers)
		Barrier();
	
	return retval;
}

template<class ScratchAccessor, bool edgeBarriers>
bool inverseBallot()
{
	return ballotBitExtract<ScratchAccessor, edgeBarriers>(gl_LocalInvocationIndex);
}

/**
 * Broadcasts the value `val` of invocation index `id`
 * to all other invocations.
 * 
 * We save the value in the shared array in the bitfieldDWORDs index 
 * and then all invocations access that index.
 * 
 * We want to broadcast types other than uint but because 
 * we use a uint scratch array, we need to convert these types 
 * into uint then back again. We use the converter template for this, 
 * which also contains the source type.
 */
template<typename T, class ScratchAccessor>
T broadcast(in T val, in uint id)
{
// REVIEW: Check if we need edge barriers
	ScratchAccessor scratch;
	if(gl_LocalInvocationIndex == id) {
		scratch.set(bitfieldDWORDs, val); // remember, the type conversion happens from the ScratchAccessor
	}
	Barrier();
	return scratch.get(bitfieldDWORDs);
}

// REVIEW: Should we have broadcastFirst and broadcastElected?
template<typename T, class ScratchAccessor>
T broadcastFirst(in T val)
{
	ScratchAccessor scratch;
	if (Elect())
		scratch.set(bitfieldDWORDs, val);
	Barrier();
	return scratch.get(bitfieldDWORDs);
}

/**
 * Gives us the sum of all ballots for the workgroup.
 *
 * Only the first few invocations are used for performing the sum 
 * since we only have `bitfieldDWORDs` amount of Uints that we need 
 * to add together.
 * 
 * We add them all in the scratch array index after the last DWORD 
 * that is used for the ballots. For example, if we have 128 workgroup size,
 * then the array index in which we accumulate the sum is `4` since 
 * indexes 0..3 are used for ballots.
 */ 
template<class ScratchAccessor>
uint ballotBitCount()
{
	ScratchAccessor scratch;
	scratch.set(bitfieldDWORDs, 0u);
	Barrier();
	if(gl_LocalInvocationIndex < bitfieldDWORDs)
	{
		const uint localBallot = scratch.get(gl_LocalInvocationIndex);
		const uint localBallotBitCount = countbits(localBallot);
		uint temp;
		scratch.atomicAdd(bitfieldDWORDs, localBallotBitCount, temp);
	}
	Barrier();
	return scratch.get(bitfieldDWORDs);
}

template<class ScratchAccessor>
uint ballotScanBitCount(in bool exclusive)
{
	ScratchAccessor scratch;
	subgroup::ScratchOffsetsAndMasks offsetsAndMasks = subgroup::ScratchOffsetsAndMasks::WithDefaults();
	const uint _dword = getDWORD(gl_LocalInvocationIndex);
	const uint localBitfield = scratch.get(_dword);
	uint globalCount;
	{
		uint localBitfieldBackup;
		if(gl_LocalInvocationIndex < bitfieldDWORDs)
		{
			localBitfieldBackup = scratch.get(gl_LocalInvocationIndex);
		}
		// scan hierarchically, invocations with `gl_LocalInvocationIndex >= bitfieldDWORDs` will have garbage here
		Barrier();
		
        using WSHT = WorkgroupScanHead<uint, subgroup::inclusive_scan<uint, binops::add<uint>, ScratchAccessor >, ScratchAccessor>;
		WSHT wsh = WSHT::create(true, 0u, bitfieldDWORDs);
		wsh();

        using WSTT = WorkgroupScanTail<uint, binops::add<uint>, ScratchAccessor>;
		WSTT wst = WSTT::create(true, 0u, wsh.lastInvocation, wsh.scanStoreIndex);
		wst();
		
		// fix it (abuse the fact memory is left over)
		globalCount = _dword != 0u ? scratch.get(_dword) : 0u;
		Barrier();
		
		// restore
		if(gl_LocalInvocationIndex < bitfieldDWORDs)
		{
			scratch.set(gl_LocalInvocationIndex, localBitfieldBackup);
		}
		Barrier();
	}
	const uint mask = (exclusive ? 0x7fFFffFFu:0xFFffFFffu)>>(31u-(gl_LocalInvocationIndex&31u));
	return globalCount + countbits(localBitfield & mask);
}

template<class ScratchAccessor>
uint ballotInclusiveBitCount()
{
	return ballotScanBitCount<ScratchAccessor>(false);
}

template<class ScratchAccessor>
uint ballotExclusiveBitCount()
{
	return ballotScanBitCount<ScratchAccessor>(true);
}

}
}
}
#endif