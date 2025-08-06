// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/transmission/lambertian.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission/smooth_dielectric.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission/beckmann.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission/ggx.hlsl"

namespace nbl
{
namespace hlsl
{

// After Clang-HLSL introduces https://en.cppreference.com/w/cpp/language/namespace_alias
// namespace bsdf = bxdf::transmission;

}
}

#endif
