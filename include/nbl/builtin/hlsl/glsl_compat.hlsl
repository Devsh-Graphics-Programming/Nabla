// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_INCLUDED_

#include "nbl/builtin/hlsl/spirv_intrinsics.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{
#pragma region BASIC
uint gl_SubgroupInvocationID() {
	// TODO (PentaKon): SPIRV Intrinsics
	return WaveGetLaneIndex();
}
#pragma endregion BASIC
	
#pragma region ATOMICS
template<typename T>
T atomicAdd(inout T ptr, T value)
{
	return spirv::atomicAdd(ptr, 1, 0, value);
}
template<typename T>
T atomicAnd(inout T ptr, T value)
{
	return spirv::atomicAnd(ptr, 1, 0, value);
}
template<typename T>
T atomicOr(inout T ptr, T value)
{
	return spirv::atomicOr(ptr, 1, 0, value);
}
template<typename T>
T atomicXor(inout T ptr, T value)
{
	return spirv::atomicXor(ptr, 1, 0, value);
}
template<typename T>
T atomicMin(inout T ptr, T value)
{
	return spirv::atomicMin(ptr, 1, 0, value);
}
template<typename T>
T atomicMax(inout T ptr, T value)
{
	return spirv::atomicMax(ptr, 1, 0, value);
}
template<typename T>
T atomicExchange(inout T ptr, T value)
{
	return spirv::atomicExchange(ptr, 1, 0, value);
}
template<typename T>
T atomicCompSwap(inout T ptr, T comparator, T value)
{
	return spirv::atomicCompSwap(ptr, 1, 0, 0, value, comparator);
}
#pragma endregion ATOMICS

#pragma region BALLOT
template<typename T>
T subgroupBroadcastFirst(T value)
{
	return spirv::subgroupBroadcastFirst(3, value);
}

template<typename T>
T subgroupBroadcast(T value, uint invocationId)
{
	return spirv::subgroupBroadcast(3, value, invocationId);
}
#pragma endregion BALLOT

#pragma region SHUFFLE
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
#pragma endregion SHUFFLE

#pragma region BARRIERS
void barrier() {
	spirv::controlBarrier(2, 2, 0x8 | 0x100);
}

void memoryBarrierShared() {
	spirv::memoryBarrier(1, 0x8 | 0x100);
}

// Memory Semantics: AcquireRelease, UniformMemory, WorkgroupMemory, AtomicCounterMemory, ImageMemory
void subgroupBarrier() {
	// REVIEW-519: Not sure what to do with this as it acts on Workgroup scope and this produces incorrect behavior. 
	// We might as well assume subgroupBarrier is simply unsupported, which makes sense since they execute in lockstep anyway.
	//spirv::controlBarrier(3, 3, 0x800 | 0x400 | 0x100 | 0x40 | 0x8);
}

void subgroupMemoryBarrierShared() {
	spirv::memoryBarrier(3, 0x800 | 0x400 | 0x100 | 0x40 | 0x8);
}
#pragma endregion BARRIERS

}
}
}

#endif