// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_BALLOT_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_SUBGROUP_BALLOT_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/ballot.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{

template<typename T>
T subgroupBroadcastFirst(T value)
{
	return spirv::ballot::subgroupBroadcastFirst<T>(/* Subgroup Scope */ 3, value);
}

template<typename T>
T subgroupBroadcast(T value, uint invocationId)
{
	return spirv::ballot::subgroupBroadcast<T>(/* Subgroup Scope */ 3, value, invocationId);
}

}
}
}

#endif