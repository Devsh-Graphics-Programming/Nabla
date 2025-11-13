// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_IRIDESCENT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_IRIDESCENT_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/reflection/ggx.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class Config>
using SIridescent = SCookTorrance<Config, ndf::GGX<typename Config::scalar_type, false, ndf::MTT_REFLECT>, fresnel::Iridescent<typename Config::spectral_type, false> >;

}

// inherit trait from cook torrance base

}
}
}

#endif
