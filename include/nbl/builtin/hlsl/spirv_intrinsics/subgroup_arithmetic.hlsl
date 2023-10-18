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

[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(349)]]
int groupAdd(uint groupScope, [[vk::ext_literal]] uint operation, int value);
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(349)]]
uint groupAdd(uint groupScope, [[vk::ext_literal]] uint operation, uint value);
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(350)]]
float groupAdd(uint groupScope, [[vk::ext_literal]] uint operation, float value);

[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(351)]]
int groupMul(uint groupScope, [[vk::ext_literal]] uint operation, int value);
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(351)]]
uint groupMul(uint groupScope, [[vk::ext_literal]] uint operation, uint value);
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(352)]]
float groupMul(uint groupScope, [[vk::ext_literal]] uint operation, float value);

template<typename T>
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(359)]]
T groupBitwiseAnd(uint groupScope, [[vk::ext_literal]] uint operation, T value);

template<typename T>
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(360)]]
T groupBitwiseOr(uint groupScope, [[vk::ext_literal]] uint operation, T value);

template<typename T>
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(361)]]
T groupBitwiseXor(uint groupScope, [[vk::ext_literal]] uint operation, T value);

// The MIN and MAX operations in SPIR-V have different Ops for each arithmetic type
// so we implement them distinctly
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(353)]]
int groupBitwiseMin(uint groupScope, [[vk::ext_literal]] uint operation, int value);
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(354)]]
uint groupBitwiseMin(uint groupScope, [[vk::ext_literal]] uint operation, uint value);
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(355)]]
float groupBitwiseMin(uint groupScope, [[vk::ext_literal]] uint operation, float value);

[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(356)]]
int groupBitwiseMax(uint groupScope, [[vk::ext_literal]] uint operation, int value);
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(357)]]
uint groupBitwiseMax(uint groupScope, [[vk::ext_literal]] uint operation, uint value);
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
[[vk::ext_instruction(358)]]
float groupBitwiseMax(uint groupScope, [[vk::ext_literal]] uint operation, float value);

}
}
}

#endif