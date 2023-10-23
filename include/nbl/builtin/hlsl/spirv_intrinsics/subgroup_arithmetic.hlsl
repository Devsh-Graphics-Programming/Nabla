// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_ARITHMETIC_INCLUDED_

// For all WaveMultiPrefix* ops, an example can be found here https://github.com/microsoft/DirectXShaderCompiler/blob/4e5440e1ee1f30d1164f90445611328293de08fa/tools/clang/test/HLSLFileCheck/hlsl/intrinsics/wave/prefix/sm_6_5_wave.hlsl
// However, we prefer to implement them with SPIRV intrinsics to avoid DXC changes in the compiler's emitted code

namespace nbl 
{
namespace hlsl
{
namespace spirv
{

[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformIAdd  )]]
int32_t groupAdd(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, int32_t value);
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformIAdd  )]]
uint32_t groupAdd(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformFAdd  )]]
float groupAdd(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformIMul )]]
int32_t groupMul(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, int32_t value);
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformIMul )]]
uint32_t groupMul(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformFMul )]]
float groupMul(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, float value);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBitwiseAnd )]]
T groupBitwiseAnd(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBitwiseOr )]]
T groupBitwiseOr(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformBitwiseXor )]]
T groupBitwiseXor(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);

// The MIN and MAX operations in SPIR-V have different Ops for each arithmetic type
// so we implement them distinctly
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformSMin )]]
int32_t groupBitwiseMin(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, int32_t value);
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformUMin )]]
uint32_t groupBitwiseMin(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformFMin )]]
float groupBitwiseMin(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, float value);

[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformSMax )]]
int32_t groupBitwiseMax(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, int32_t value);
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformUMax )]]
uint32_t groupBitwiseMax(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, uint32_t value);
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformFMax )]]
float groupBitwiseMax(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, float value);

}
}
}

#endif
