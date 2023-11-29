// Copyright (C) 2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_DEVICE_CAPABILITIES_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_DEVICE_CAPABILITIES_TRAITS_INCLUDED_

#include <nbl/builtin/hlsl/member_test_macros.hlsl>

#ifdef __HLSL_VERSION
namespace nbl
{
namespace hlsl
{
template<typename device_capabilities>
struct device_capabilities_traits
{
    // TODO: check for members and default them to sane things, only do the 5 members in CJITIncludeLoader.cpp struct, we'll do the rest on `vulkan_1_3` branch with Nahim
};
}
}
#endif
#endif