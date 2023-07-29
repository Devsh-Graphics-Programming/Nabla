// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SUBGROUP_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SUBGROUP_SHUFFLE_INCLUDED_

#include "nbl/builtin/hlsl/subgroup/basic.hlsl"

namespace nbl
{
namespace hlsl
{
namespace subgroup
{
template<typename T>
T Shuffle(T value, uint index)
{
	return WaveReadLaneAt(value, index);
}

template<typename T>
T ShuffleUp(T value, uint delta)
{
  return Shuffle<T>(value, InvocationID() - delta);
}

template<typename T>
T ShuffleDown(T value, uint delta)
{
  return Shuffle<T>(value, InvocationID() + delta);
}

}
}
}

#endif