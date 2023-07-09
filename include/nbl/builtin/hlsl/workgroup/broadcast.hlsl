// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_WORKGROUP_BROADCAST_INCLUDED_
#define _NBL_BUILTIN_HLSL_WORKGROUP_BROADCAST_INCLUDED_

#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace workgroup
{

/**
 * Broadcasts the value `val` of invocation index `id`
 * to all other invocations.
 * 
 * We save the value in the shared array in the bitfieldDWORDs index 
 * and then all invocations access that index.
 */
template<typename T, class ScratchAccessor>
T Broadcast(in T val, in uint id)
{
// REVIEW: Check if we need edge barriers
	ScratchAccessor scratch;
	
	if(gl_LocalInvocationIndex == id) {
		scratch.broadcast.set(bitfieldDWORDs, val);
	}
	Barrier();
	return scratch.broadcast.get(bitfieldDWORDs);
}

// REVIEW: Should we have broadcastFirst and broadcastElected?
template<typename T, class ScratchAccessor>
T BroadcastFirst(in T val)
{
	ScratchAccessor scratch;
	if (Elect())
		scratch.broadcast.set(bitfieldDWORDs, val);
	Barrier();
	return scratch.broadcast.get(bitfieldDWORDs);
}

}
}
}
#endif