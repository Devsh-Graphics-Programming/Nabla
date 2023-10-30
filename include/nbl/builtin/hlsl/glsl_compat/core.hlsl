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

<<<<<<< HEAD
template<typename T>
T atomicAdd(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicAdd<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
=======
#ifdef __HLSL_VERSION
template<typename T>
T atomicAdd(NBL_REF_ARG(T) ptr, T value)
{
    return spirv::atomicAdd<T>(ptr, 1, 0, value);
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
}
template<typename T>
T atomicAnd(NBL_REF_ARG(T) ptr, T value)
{
<<<<<<< HEAD
    return spirv::atomicAnd<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
=======
    return spirv::atomicAnd<T>(ptr, 1, 0, value);
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
}
template<typename T>
T atomicOr(NBL_REF_ARG(T) ptr, T value)
{
<<<<<<< HEAD
    return spirv::atomicOr<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
=======
    return spirv::atomicOr<T>(ptr, 1, 0, value);
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
}
template<typename T>
T atomicXor(NBL_REF_ARG(T) ptr, T value)
{
<<<<<<< HEAD
    return spirv::atomicXor<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
=======
    return spirv::atomicXor<T>(ptr, 1, 0, value);
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
}
template<typename T>
T atomicMin(NBL_REF_ARG(T) ptr, T value)
{
<<<<<<< HEAD
    return spirv::atomicMin<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
=======
    return spirv::atomicMin<T>(ptr, 1, 0, value);
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
}
template<typename T>
T atomicMax(NBL_REF_ARG(T) ptr, T value)
{
<<<<<<< HEAD
    return spirv::atomicMax<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
=======
    return spirv::atomicMax<T>(ptr, 1, 0, value);
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
}
template<typename T>
T atomicExchange(NBL_REF_ARG(T) ptr, T value)
{
<<<<<<< HEAD
    return spirv::atomicExchange<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, value);
=======
    return spirv::atomicExchange<T>(ptr, 1, 0, value);
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
}
template<typename T>
T atomicCompSwap(NBL_REF_ARG(T) ptr, T comparator, T value)
{
<<<<<<< HEAD
    return spirv::atomicCompSwap<T>(ptr, spv::ScopeDevice, spv::GroupOperationReduce, spv::GroupOperationReduce, value, comparator);
}

void barrier() {
    spirv::controlBarrier(spv::ScopeDevice, spv::GroupOperationReduce, 0x8 | 0x100);
}

void memoryBarrierShared() {
    spirv::memoryBarrier(spv::ScopeDevice, 0x8 | 0x100);
}
=======
    return spirv::atomicCompSwap<T>(ptr, 1, 0, 0, value, comparator);
}

/**
 * For Compute Shaders
 */

// TODO (Future): Its annoying we have to forward declare those, but accessing gl_NumSubgroups and other gl_* values is not yet possible due to https://github.com/microsoft/DirectXShaderCompiler/issues/4217
// also https://github.com/microsoft/DirectXShaderCompiler/issues/5280
uint32_t gl_LocalInvocationIndex();
uint32_t3 gl_WorkGroupSize();
uint32_t3 gl_GlobalInvocationID();
uint32_t3 gl_WorkGroupID();

void barrier() {
    spirv::controlBarrier(spv::ScopeWorkgroup, spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask);
}

/**
 * For Tessellation Control Shaders
 */
void tess_ctrl_barrier() {
    spirv::controlBarrier(spv::ScopeWorkgroup, spv::ScopeInvocation, 0);
}

void memoryBarrierShared() {
    spirv::memoryBarrier(spv::ScopeDevice, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask);
}
#endif
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8

}
}
}

#endif