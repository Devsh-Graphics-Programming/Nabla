// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_FRAGMENT_SHADER_BARYCENTRIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_SPIRV_INTRINSICS_FRAGMENT_SHADER_BARYCENTRIC_INCLUDED_

#include "spirv/unified1/spirv.hpp"

namespace nbl 
{
namespace hlsl
{
namespace spirv
{

[[vk::ext_capability(/*spv::CapabilityFragmentBarycentricKHR*/5284)]]
[[vk::ext_extension("SPV_KHR_fragment_shader_barycentric")]]
[[vk::ext_builtin_input(/*spv::BuiltInBaryCoordKHR*/5286)]]
static const float32_t3 BaryCoordKHR;

[[vk::ext_capability(/*spv::CapabilityFragmentBarycentricKHR*/5284)]]
[[vk::ext_extension("SPV_KHR_fragment_shader_barycentric")]]
[[vk::ext_builtin_input(/*spv::BuiltInBaryCoordKHR*/5287)]]
static const float32_t3 BaryCoordNoPerspKHR;

}
}
}

#endif
