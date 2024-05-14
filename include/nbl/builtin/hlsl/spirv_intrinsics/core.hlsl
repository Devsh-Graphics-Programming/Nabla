// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_CORE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_CORE_INCLUDED_


#ifdef __HLSL_VERSION // TODO: AnastZIuk fix public search paths so we don't choke
#include "spirv/unified1/spirv.hpp"
#include "spirv/unified1/GLSL.std.450.h"
#endif


namespace nbl 
{
namespace hlsl
{
#ifdef __HLSL_VERSION
namespace spirv
{
//! General
[[vk::ext_builtin_input(spv::BuiltInHelperInvocation)]]
static const bool HelperInvocation;

//! Vertex Inputs
[[vk::ext_builtin_input(spv::BuiltInVertexIndex)]]
static const uint32_t VertexIndex;
[[vk::ext_builtin_input(spv::BuiltInInstanceIndex)]]
static const uint32_t InstanceIndex;

//! Vertex and friends
[[vk::ext_builtin_output(spv::BuiltInPosition)]]
static float32_t4 Position;

//! Compute Shader Builtins
[[vk::ext_builtin_input(spv::BuiltInNumWorkgroups)]]
static const uint32_t3 NumWorkGroups;
// TODO: Doesn't work, find out why and file issue on DXC!
//[[vk::ext_builtin_input(spv::BuiltInWorkgroupSize)]]
//static const uint32_t3 WorkgroupSize;
[[vk::ext_builtin_input(spv::BuiltInWorkgroupId)]]
static const uint32_t3 WorkgroupId;
[[vk::ext_builtin_input(spv::BuiltInLocalInvocationId)]]
static const uint32_t3 LocalInvocationId;
[[vk::ext_builtin_input(spv::BuiltInGlobalInvocationId)]]
static const uint32_t3 GlobalInvocationId;
[[vk::ext_builtin_input(spv::BuiltInLocalInvocationIndex)]]
static const uint32_t LocalInvocationIndex;

//! General Operations

// Here's the thing with atomics, it's not only the data type that dictates whether you can do an atomic or not.
// It's the storage class that has the most effect (shared vs storage vs image) and we can't check that easily
template<typename T> // integers operate on 2s complement so same op for signed and unsigned
[[vk::ext_instruction(spv::OpAtomicIAdd)]]
T atomicIAdd([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicAnd)]]
T atomicAnd([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicOr)]]
T atomicOr([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicXor)]]
T atomicXor([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename Signed>
[[vk::ext_instruction( spv::OpAtomicSMin )]]
Signed atomicSMin([[vk::ext_reference]] int32_t ptr, uint32_t memoryScope, uint32_t memorySemantics, Signed value);
template<typename Unsigned>
[[vk::ext_instruction( spv::OpAtomicUMin )]]
Unsigned atomicUMin([[vk::ext_reference]] Unsigned ptr, uint32_t memoryScope, uint32_t memorySemantics, Unsigned value);

template<typename Signed>
[[vk::ext_instruction( spv::OpAtomicSMax )]]
Signed atomicSMax([[vk::ext_reference]] Signed ptr, uint32_t memoryScope, uint32_t memorySemantics, Signed value);
template<typename Unsigned>
[[vk::ext_instruction( spv::OpAtomicUMax )]]
Unsigned atomicUMax([[vk::ext_reference]] uint32_t ptr, uint32_t memoryScope, uint32_t memorySemantics, Unsigned value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicExchange)]]
T atomicExchange([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicCompareExchange)]]
T atomicCompareExchange([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memSemanticsEqual, uint32_t memSemanticsUnequal, T value, T comparator);

//! Std 450 Extended set operations
template<typename SquareMatrix>
[[vk::ext_instruction(GLSLstd450MatrixInverse)]]
SquareMatrix matrixInverse(NBL_CONST_REF_ARG(SquareMatrix) mat);

// Memory Semantics link here: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Memory_Semantics_-id-

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_memory_semantics_id
// By providing memory semantics None we do both control and memory barrier as is done in GLSL

[[vk::ext_instruction( spv::OpControlBarrier )]]
void controlBarrier(uint32_t executionScope, uint32_t memoryScope, uint32_t memorySemantics);

[[vk::ext_instruction( spv::OpMemoryBarrier )]]
void memoryBarrier(uint32_t memoryScope, uint32_t memorySemantics);


// Add specializations if you need to emit a `ext_capability` (this means that the instruction needs to forward through an `impl::` struct and so on)
template<class T, class U>
[[vk::ext_instruction(spv::OpBitcast)]]
T bitcast(U);

template<typename Unsigned>
[[vk::ext_instruction( spv::OpBitFieldUExtract )]]
Unsigned bitFieldUExtract( Unsigned val, uint32_t offsetBits, uint32_t numBits );

template<typename Signed>
[[vk::ext_instruction( spv::OpBitFieldSExtract )]]
Signed bitFieldSExtract( Signed val, uint32_t offsetBits, uint32_t numBits );

}
#endif
}
}

#endif