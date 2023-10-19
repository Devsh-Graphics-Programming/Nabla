// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_CORE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_CORE_INCLUDED_


#include "spirv/1.2/spirv.h"


namespace nbl 
{
namespace hlsl
{
namespace spirv
{

#pragma region ATOMICS
template<typename T>
T atomicAdd([[vk::ext_reference]] T ptr, uint memoryScope, uint memorySemantics, T value);
template<>
[[vk::ext_instruction(SpvOpAtomicIAdd)]]
int atomicAdd([[vk::ext_reference]] int ptr, uint memoryScope, uint memorySemantics, int value);
template<>
[[vk::ext_instruction(/* OpAtomicIAdd */ 234)]]
uint atomicAdd([[vk::ext_reference]] uint ptr, uint memoryScope, uint memorySemantics, uint value);

template<typename T>
T atomicAnd([[vk::ext_reference]] T ptr, uint memoryScope, uint memorySemantics, T value);
template<>
[[vk::ext_instruction(/* OpAtomicAnd */ 240)]]
int atomicAnd([[vk::ext_reference]] int ptr, uint memoryScope, uint memorySemantics, int value);
template<>
[[vk::ext_instruction(/* OpAtomicAnd */ 240)]]
uint atomicAnd([[vk::ext_reference]] uint ptr, uint memoryScope, uint memorySemantics, uint value);

template<typename T>
T atomicOr([[vk::ext_reference]] T ptr, uint memoryScope, uint memorySemantics, T value);
template<>
[[vk::ext_instruction(/* OpAtomicOr */ 241)]]
int atomicOr([[vk::ext_reference]] int ptr, uint memoryScope, uint memorySemantics, int value);
template<>
[[vk::ext_instruction(/* OpAtomicOr */ 241)]]
uint atomicOr([[vk::ext_reference]] uint ptr, uint memoryScope, uint memorySemantics, uint value);

template<typename T>
T atomicXor([[vk::ext_reference]] T ptr, uint memoryScope, uint memorySemantics, T value);
template<>
[[vk::ext_instruction(/* OpAtomicXor */ 242)]]
int atomicXor([[vk::ext_reference]] int ptr, uint memoryScope, uint memorySemantics, int value);
template<>
[[vk::ext_instruction(/* OpAtomicXor */ 242)]]
uint atomicXor([[vk::ext_reference]] uint ptr, uint memoryScope, uint memorySemantics, uint value);

template<typename T>
T atomicMin([[vk::ext_reference]] T ptr, uint memoryScope, uint memorySemantics, T value);
template<>
[[vk::ext_instruction(/* OpAtomicSMin */ 236)]]
int atomicMin([[vk::ext_reference]] int ptr, uint memoryScope, uint memorySemantics, int value);
template<>
[[vk::ext_instruction(/* OpAtomicSMin */ 236)]]
uint atomicMin([[vk::ext_reference]] uint ptr, uint memoryScope, uint memorySemantics, uint value);

template<typename T>
T atomicMax([[vk::ext_reference]] T ptr, uint memoryScope, uint memorySemantics, T value);
template<>
[[vk::ext_instruction(/* OpAtomicSMax */ 238)]]
int atomicMax([[vk::ext_reference]] int ptr, uint memoryScope, uint memorySemantics, int value);
template<>
[[vk::ext_instruction(/* OpAtomicSMax */ 238)]]
uint atomicMax([[vk::ext_reference]] uint ptr, uint memoryScope, uint memorySemantics, uint value);

template<typename T>
T atomicExchange([[vk::ext_reference]] T ptr, uint memoryScope, uint memorySemantics, T value);
template<>
[[vk::ext_instruction(/* OpAtomicExchange */ 229)]]
int atomicExchange([[vk::ext_reference]] int ptr, uint memoryScope, uint memorySemantics, int value);
template<>
[[vk::ext_instruction(/* OpAtomicExchange */ 229)]]
uint atomicExchange([[vk::ext_reference]] uint ptr, uint memoryScope, uint memorySemantics, uint value);
template<>
[[vk::ext_instruction(/* OpAtomicExchange */ 229)]]
float atomicExchange([[vk::ext_reference]] float ptr, uint memoryScope, uint memorySemantics, float value);


template<typename T>
T atomicCompSwap([[vk::ext_reference]] T ptr, uint memoryScope, uint memSemanticsEqual, uint memSemanticsUnequal, T value, T comparator);
template<>
[[vk::ext_instruction(/* OpAtomicCompareExchange */ 230)]]
int atomicCompSwap([[vk::ext_reference]] int ptr, uint memoryScope, uint memSemanticsEqual, uint memSemanticsUnequal, int value, int comparator);
template<>
[[vk::ext_instruction(/* OpAtomicCompareExchange */ 230)]]
uint atomicCompSwap([[vk::ext_reference]] uint ptr, uint memoryScope, uint memSemanticsEqual, uint memSemanticsUnequal, uint value, uint comparator);
#pragma endregion ATOMICS

#pragma region BARRIERS
// Memory Semantics link here: https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#Memory_Semantics_-id-

// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_memory_semantics_id
// By providing memory semantics None we do both control and memory barrier as is done in GLSL

[[vk::ext_instruction(/* OpControlBarrier */ 224)]]
void controlBarrier(uint executionScope, uint memoryScope, uint memorySemantics);

[[vk::ext_instruction(/* OpMemoryBarrier */ 225)]]
void memoryBarrier(uint memoryScope, uint memorySemantics);
#pragma endregion BARRIERS
}
}
}

#endif