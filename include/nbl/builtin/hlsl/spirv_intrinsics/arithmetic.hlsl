// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_INTRINSICS_ARITHMETIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_INTRINSICS_ARITHMETIC_INCLUDED_

// For all WaveMultiPrefix* ops, an example can be found here https://github.com/microsoft/DirectXShaderCompiler/blob/4e5440e1ee1f30d1164f90445611328293de08fa/tools/clang/test/HLSLFileCheck/hlsl/intrinsics/wave/prefix/sm_6_5_wave.hlsl
// However, we prefer to implement them with SPIRV intrinsics to avoid DXC changes in the compiler's emitted code

namespace nbl 
{
namespace hlsl
{
namespace spirv
{
namespace impl
{
[[vk::ext_capability(/* GroupNonUniformArithmetic */ 63)]]
void arithmetic_extcap(){}
}

namespace arithmetic
{
template<typename T>
[[vk::ext_instruction(359)]]
T subgroupPrefixAnd(uint scope, [[vk::ext_literal]] uint operation, T value);

template<typename T>
[[vk::ext_instruction(360)]]
T subgroupPrefixOr(uint scope, [[vk::ext_literal]] uint operation, T value);

template<typename T>
[[vk::ext_instruction(361)]]
T subgroupPrefixXor(uint scope, [[vk::ext_literal]] uint operation, T value);

// The MIN and MAX operations in SPIR-V have different Ops for each arithmetic type
// so we implement them distinctly
[[vk::ext_instruction(353)]]
int subgroupPrefixMin(uint scope, [[vk::ext_literal]] uint operation, int value);
[[vk::ext_instruction(354)]]
uint subgroupPrefixMin(uint scope, [[vk::ext_literal]] uint operation, uint value);
[[vk::ext_instruction(355)]]
float subgroupPrefixMin(uint scope, [[vk::ext_literal]] uint operation, float value);

[[vk::ext_instruction(356)]]
int subgroupPrefixMax(uint scope, [[vk::ext_literal]] uint operation, int value);
[[vk::ext_instruction(357)]]
uint subgroupPrefixMax(uint scope, [[vk::ext_literal]] uint operation, uint value);
[[vk::ext_instruction(358)]]
float subgroupPrefixMax(uint scope, [[vk::ext_literal]] uint operation, float value);

}

}
}
}

#endif