// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_GLSL_COMPAT_CORE_INCLUDED_
#define _NBL_BUILTIN_HLSL_GLSL_COMPAT_CORE_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace glsl
{

template<typename T>
T atomicAdd(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicAdd<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
}
template<typename T>
T atomicAnd(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicAnd<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
}
template<typename T>
T atomicOr(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicOr<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
}
template<typename T>
T atomicXor(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicXor<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
}
template<typename T>
T atomicMin(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicMin<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
}
template<typename T>
T atomicMax(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicMax<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
}
template<typename T>
T atomicExchange(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicExchange<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
}
template<typename T>
T atomicCompSwap(NBL_REF_ARG(T) ptr, T comparator, T value)
{
    return spirv::atomicCompSwap<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, spv::GroupOperationReduce, value, comparator);
}

void barrier() {
    spirv::controlBarrier(spv::ScopeDevice, spv::GroupOperationReduce, 0x8 | 0x100);
}

void memoryBarrierShared() {
    spirv::memoryBarrier(spv::ScopeDevice, 0x8 | 0x100);
}

}
}
}

#endif