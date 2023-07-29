// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/basic.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace subgroup
{
template<typename T>
T BroadcastFirst(T value)
{
	return WaveReadLaneFirst(value);
}
	
template<typename T>
T Broadcast(T value, uint invocationId) {
	return WaveReadLaneAt(value, invocationId);
}

uint ElectedSubgroupInvocationID() {
	return BroadcastFirst<uint>(InvocationID());
}

uint ElectedLocalInvocationID() {
	return BroadcastFirst<uint>(gl_LocalInvocationIndex);
}

}
}
}

#endif