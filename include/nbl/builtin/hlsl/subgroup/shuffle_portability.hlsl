// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_SHUFFLE_PORTABILITY_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_SHUFFLE_PORTABILITY_INCLUDED_

#include <nbl/builtin/hlsl/subgroup/basic_portability.hlsl>

#ifndef _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
#define _NBL_GL_LOCAL_INVOCATION_IDX_DECLARED_
const uint gl_LocalInvocationIndex : SV_GroupIndex;
#endif

namespace nbl
{
namespace hlsl
{
namespace subgroup
{
namespace native
{
	template<typename T>
	T Shuffle(T value, uint index)
	{
		return WaveReadLaneAt(value, index);
	}
}

#ifndef NBL_GL_KHR_shader_subgroup_shuffle
namespace portability
{
	template<typename T, class ScratchAccessor>
	T Shuffle(T value, uint index)
	{
		ScratchAccessor scratch;
		scratch.set(gl_LocalInvocationIndex, value);
		Barrier();
		return scratch.get(index); // placeholder
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