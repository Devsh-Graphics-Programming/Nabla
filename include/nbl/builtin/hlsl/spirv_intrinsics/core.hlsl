// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_CORE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_CORE_INCLUDED_


#ifdef __HLSL_VERSION // TODO: AnastZIuk fix public search paths so we don't choke
#include "spirv/unified1/spirv.hpp"
#include "spirv/unified1/GLSL.std.450.h"
#endif

#include "nbl/builtin/hlsl/type_traits.hlsl"

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

//! General Types
template<uint32_t StorageClass, typename T>
using pointer_t = vk::SpirvOpaqueType<spv::OpTypePointer,vk::Literal<vk::integral_constant<uint32_t,StorageClass> >,T>;

//! General Operations

// The holy operation that makes addrof possible
template<uint32_t StorageClass, typename T>
[[vk::ext_instruction(spv::OpCopyObject)]]
pointer_t<StorageClass,T> copyObject([[vk::ext_reference]] T v);

// Here's the thing with atomics, it's not only the data type that dictates whether you can do an atomic or not.
// It's the storage class that has the most effect (shared vs storage vs image) and we can't check that easily
template<typename T> // integers operate on 2s complement so same op for signed and unsigned
[[vk::ext_instruction(spv::OpAtomicIAdd)]]
enable_if_t<is_same_v<T,uint32_t> || is_same_v<T,int32_t>, T> atomicIAdd([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicIAdd)]]
enable_if_t<is_spirv_type_v<Ptr_T> && (is_same_v<T,uint32_t> || is_same_v<T,int32_t>), T> atomicIAdd(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T> // integers operate on 2s complement so same op for signed and unsigned
[[vk::ext_capability(spv::CapabilityInt64Atomics)]]
[[vk::ext_instruction(spv::OpAtomicIAdd)]]
enable_if_t<is_same_v<T,uint64_t> || is_same_v<T,int64_t>, T> atomicIAdd([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T, typename Ptr_T> // DXC Workaround
[[vk::ext_capability(spv::CapabilityInt64Atomics)]]
[[vk::ext_instruction(spv::OpAtomicIAdd)]]
enable_if_t<is_spirv_type_v<Ptr_T> && (is_same_v<T,uint64_t> || is_same_v<T,int64_t>), T> atomicIAdd(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T> // integers operate on 2s complement so same op for signed and unsigned
[[vk::ext_instruction(spv::OpAtomicISub)]]
enable_if_t<is_same_v<T,uint32_t> || is_same_v<T,int32_t>, T> atomicISub([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicISub)]]
enable_if_t<is_spirv_type_v<Ptr_T> && (is_same_v<T,uint32_t> || is_same_v<T,int32_t>), T> atomicISub(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T> // integers operate on 2s complement so same op for signed and unsigned
[[vk::ext_capability(spv::CapabilityInt64Atomics)]]
[[vk::ext_instruction(spv::OpAtomicISub)]]
enable_if_t<is_same_v<T,uint64_t> || is_same_v<T,int64_t>, T> atomicISub([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T, typename Ptr_T> // DXC Workaround
[[vk::ext_capability(spv::CapabilityInt64Atomics)]]
[[vk::ext_instruction(spv::OpAtomicISub)]]
enable_if_t<is_spirv_type_v<Ptr_T> && (is_same_v<T,uint64_t> || is_same_v<T,int64_t>), T> atomicISub(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicAnd)]]
enable_if_t<is_same_v<T,uint32_t> || is_same_v<T,int32_t>, T> atomicAnd([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicAnd)]]
enable_if_t<is_spirv_type_v<Ptr_T> && (is_same_v<T,uint32_t> || is_same_v<T,int32_t>), T> atomicAnd(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicOr)]]
enable_if_t<is_same_v<T,uint32_t> || is_same_v<T,int32_t>, T> atomicOr([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicOr)]]
enable_if_t<is_spirv_type_v<Ptr_T> && (is_same_v<T,uint32_t> || is_same_v<T,int32_t>), T> atomicOr(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicXor)]]
enable_if_t<is_same_v<T,uint32_t> || is_same_v<T,int32_t>, T> atomicXor([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicXor)]]
enable_if_t<is_spirv_type_v<Ptr_T> && (is_same_v<T,uint32_t> || is_same_v<T,int32_t>), T> atomicXor(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename Signed>
[[vk::ext_instruction( spv::OpAtomicSMin )]]
enable_if_t<is_same_v<Signed,int32_t>, Signed> atomicSMin([[vk::ext_reference]] int32_t ptr, uint32_t memoryScope, uint32_t memorySemantics, Signed value);

template<typename Signed, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicSMin)]]
enable_if_t<is_spirv_type_v<Ptr_T> && is_same_v<Signed,int32_t>, Signed> atomicSMin(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, Signed value);

template<typename Unsigned>
[[vk::ext_instruction( spv::OpAtomicUMin )]]
enable_if_t<is_same_v<Unsigned,uint32_t>, Unsigned> atomicUMin([[vk::ext_reference]] Unsigned ptr, uint32_t memoryScope, uint32_t memorySemantics, Unsigned value);

template<typename Unsigned, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicUMin)]]
enable_if_t<is_spirv_type_v<Ptr_T> && is_same_v<Unsigned,uint32_t>, Unsigned> atomicUMin(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, Unsigned value);

template<typename Signed>
[[vk::ext_instruction( spv::OpAtomicSMax )]]
enable_if_t<is_same_v<Signed,int32_t>, Signed> atomicSMax([[vk::ext_reference]] Signed ptr, uint32_t memoryScope, uint32_t memorySemantics, Signed value);

template<typename Signed, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicSMax)]]
enable_if_t<is_spirv_type_v<Ptr_T> && is_same_v<Signed,int32_t>, Signed> atomicSMax(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, Signed value);

template<typename Unsigned>
[[vk::ext_instruction( spv::OpAtomicUMax )]]
enable_if_t<is_same_v<Unsigned,uint32_t>, Unsigned> atomicUMax([[vk::ext_reference]] uint32_t ptr, uint32_t memoryScope, uint32_t memorySemantics, Unsigned value);

template<typename Unsigned, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicUMax)]]
enable_if_t<is_spirv_type_v<Ptr_T> && is_same_v<Unsigned,uint32_t>, Unsigned> atomicUMax(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, Unsigned value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicExchange)]]
T atomicExchange([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicExchange)]]
enable_if_t<is_spirv_type_v<Ptr_T>, T> atomicExchange(Ptr_T ptr, uint32_t memoryScope, uint32_t memorySemantics, T value);

template<typename T>
[[vk::ext_instruction(spv::OpAtomicCompareExchange)]]
T atomicCompareExchange([[vk::ext_reference]] T ptr, uint32_t memoryScope, uint32_t memSemanticsEqual, uint32_t memSemanticsUnequal, T value, T comparator);

template<typename T, typename Ptr_T> // DXC Workaround
[[vk::ext_instruction(spv::OpAtomicCompareExchange)]]
enable_if_t<is_spirv_type_v<Ptr_T>, T> atomicCompareExchange(Ptr_T ptr, uint32_t memoryScope, uint32_t memSemanticsEqual, uint32_t memSemanticsUnequal, T value, T comparator);


template<typename T, uint32_t alignment>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpLoad)]]
T load(pointer_t<spv::StorageClassPhysicalStorageBuffer,T> pointer, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpLoad)]]
enable_if_t<is_spirv_type_v<P>,T> load(P pointer);

template<typename T, uint32_t alignment>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpStore)]]
void store(pointer_t<spv::StorageClassPhysicalStorageBuffer,T>  pointer, T obj, [[vk::ext_literal]] uint32_t __aligned = /*Aligned*/0x00000002, [[vk::ext_literal]] uint32_t __alignment = alignment);

template<typename T, typename P>
[[vk::ext_instruction(spv::OpStore)]]
enable_if_t<is_spirv_type_v<P>,void> store(P pointer, T obj);

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
template<typename T, typename U>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpBitcast)]]
enable_if_t<is_spirv_type_v<T> && is_spirv_type_v<U>, T> bitcast(U);

template<typename T>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpBitcast)]]
uint64_t bitcast(pointer_t<spv::StorageClassPhysicalStorageBuffer,T>);

template<typename T>
[[vk::ext_capability(spv::CapabilityPhysicalStorageBufferAddresses)]]
[[vk::ext_instruction(spv::OpBitcast)]]
pointer_t<spv::StorageClassPhysicalStorageBuffer,T> bitcast(uint64_t);

template<class T, class U>
[[vk::ext_instruction(spv::OpBitcast)]]
T bitcast(U);

template<typename Unsigned>
[[vk::ext_instruction( spv::OpBitFieldUExtract )]]
enable_if_t<is_unsigned_v<Unsigned>, Unsigned> bitFieldUExtract( Unsigned val, uint32_t offsetBits, uint32_t numBits );

template<typename Signed>
[[vk::ext_instruction( spv::OpBitFieldSExtract )]]
enable_if_t<is_signed_v<Signed>, Signed> bitFieldSExtract( Signed val, uint32_t offsetBits, uint32_t numBits );

template<typename Integral>
[[vk::ext_instruction( spv::OpBitFieldInsert )]]
Integral bitFieldInsert( Integral base, Integral insert, uint32_t offset, uint32_t count );

}

#endif
    }
}

#endif