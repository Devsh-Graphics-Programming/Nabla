// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_SHUFFLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_SHUFFLE_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics/shuffle.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{
namespace shuffle
{
template<typename T>
T subgroupShuffle(T value, uint invocationId)
{
	return spirv::subgroupShuffle(3, value, invocationId);
}

template<typename T>
T subgroupShuffleUp(T value, uint delta)
{
#ifdef NBL_GL_KHR_shader_subgroup_shuffle_relative
	return spirv::subgroupShuffleUp(3, value, delta);
#else
	return spirv::subgroupShuffle(3, value, gl_SubgroupInvocationID() - delta);
#endif
}

template<typename T>
T subgroupShuffleDown(T value, uint delta)
{
#ifdef NBL_GL_KHR_shader_subgroup_shuffle_relative
	return spirv::subgroupShuffleDown(3, value, delta);
#else
	return spirv::subgroupShuffle(3, value, gl_SubgroupInvocationID() + delta);
#endif
}

}
}
}
}

#endif