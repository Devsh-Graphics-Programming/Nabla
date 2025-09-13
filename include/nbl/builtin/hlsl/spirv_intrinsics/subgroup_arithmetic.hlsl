// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_SUBGROUP_ARITHMETIC_INCLUDED_

// For all WaveMultiPrefix* ops, an example can be found here https://github.com/microsoft/DirectXShaderCompiler/blob/4e5440e1ee1f30d1164f90445611328293de08fa/tools/clang/test/HLSLFileCheck/hlsl/intrinsics/wave/prefix/sm_6_5_wave.hlsl
// However, we prefer to implement them with SPIRV intrinsics to avoid DXC changes in the compiler's emitted code


#include "nbl/builtin/hlsl/spirv_intrinsics/core.hlsl"

namespace nbl 
{
namespace hlsl
{
namespace spirv
{

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformIAdd  )]]
enable_if_t<!is_matrix_v<T> && is_integral_v<typename vector_traits<T>::scalar_type>, T> groupAdd(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);
template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformFAdd  )]]
enable_if_t<!is_matrix_v<T> && is_floating_point_v<typename vector_traits<T>::scalar_type>, T> groupAdd(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformIMul )]]
enable_if_t<!is_matrix_v<T> && is_integral_v<typename vector_traits<T>::scalar_type>, T> groupMul(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);
template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformFMul )]]
enable_if_t<!is_matrix_v<T> && is_floating_point_v<typename vector_traits<T>::scalar_type>, T> groupMul(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);

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
template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformSMin )]]
enable_if_t<!is_matrix_v<T> && is_signed_v<T> && is_integral_v<typename vector_traits<T>::scalar_type>, T> groupSMin(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);
template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformUMin )]]
enable_if_t<!is_matrix_v<T> && !is_signed_v<T> && is_integral_v<typename vector_traits<T>::scalar_type>, T> groupUMin(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);
template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformFMin )]]
enable_if_t<!is_matrix_v<T> && is_floating_point_v<typename vector_traits<T>::scalar_type>, T> groupFMin(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);

template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformSMax )]]
enable_if_t<!is_matrix_v<T> && is_signed_v<T> && is_integral_v<typename vector_traits<T>::scalar_type>, T> groupSMax(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);
template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformUMax )]]
enable_if_t<!is_matrix_v<T> && !is_signed_v<T> && is_integral_v<typename vector_traits<T>::scalar_type>, T> groupUMax(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);
template<typename T>
[[vk::ext_capability( spv::CapabilityGroupNonUniformArithmetic )]]
[[vk::ext_instruction( spv::OpGroupNonUniformFMax )]]
enable_if_t<!is_matrix_v<T> && is_floating_point_v<typename vector_traits<T>::scalar_type>, T> groupFMax(uint32_t groupScope, [[vk::ext_literal]] uint32_t operation, T value);

}
}
}

#endif
