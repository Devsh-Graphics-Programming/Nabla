// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_SHUFFLE_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_SHUFFLE_PORTABILITY_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/basic_portability.hlsl"

namespace nbl
{
namespace hlsl
{
namespace subgroup
{
#ifdef NBL_GL_KHR_shader_subgroup_shuffle
namespace native
{
	template<typename T>
	T Shuffle(T value, uint index)
	{
		return WaveReadLaneAt(value, index);
	}
}
#else
namespace portability
{
	// REVIEW: 	This shuffling will overwrite data in scratch that is used for the larger operation
	//			except if we save it in a temp...
	template<typename T, class ScratchAccessor>
	T Shuffle(T value, uint index)
	{
		ScratchAccessor scratch;
		// safekeep existing value in temp since scratch is 
		// probably used for something else while this is called
		T temp = scratch.shuffle.get(gl_LocalInvocationIndex);
		scratch.shuffle.set(gl_LocalInvocationIndex, value);
		Barrier();
		MemoryBarrierShared();
		T result = scratch.shuffle.get(ElectedLocalInvocationID() + index); // the shuffle offset must be based on the gl_LocalInvocationIndex of the elected subgroup invocation
		scratch.shuffle.set(gl_LocalInvocationIndex, temp); // reset previous value
		return result;
	}
}
#endif

template<typename T, class ScratchAccessor>
T Shuffle(T value, uint index)
{
#ifdef NBL_GL_KHR_shader_subgroup_shuffle
	return native::Shuffle<T>(value, index);
#else
	return portability::Shuffle<T, ScratchAccessor>(value, index);
#endif
}

template<typename T, class ScratchAccessor>
T ShuffleUp(T value, uint delta)
{
  return Shuffle<T, ScratchAccessor>(value, InvocationID() - delta);
}

template<typename T, class ScratchAccessor>
T ShuffleDown(T value, uint delta)
{
  return Shuffle<T, ScratchAccessor>(value, InvocationID() + delta);
}

}
}
}

#endif