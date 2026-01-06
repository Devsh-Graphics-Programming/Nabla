// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_GGX_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/base/cook_torrance_base.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class Config>
using SGGXDielectricIsotropic = SCookTorrance<Config, ndf::GGX<typename Config::scalar_type, false, ndf::MTT_REFLECT_REFRACT>, fresnel::Dielectric<typename Config::monochrome_type> >;

template<class Config>
using SGGXDielectricAnisotropic = SCookTorrance<Config, ndf::GGX<typename Config::scalar_type, true, ndf::MTT_REFLECT_REFRACT>, fresnel::Dielectric<typename Config::monochrome_type> >;

}

// inherit trait from cook torrance base

}
}
}

#endif
