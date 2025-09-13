// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/reflection/lambertian.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection/oren_nayar.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection/beckmann.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection/ggx.hlsl"

namespace nbl
{
namespace hlsl
{

// After Clang-HLSL introduces https://en.cppreference.com/w/cpp/language/namespace_alias
// namespace brdf = bxdf::reflection;

}
}

#endif
