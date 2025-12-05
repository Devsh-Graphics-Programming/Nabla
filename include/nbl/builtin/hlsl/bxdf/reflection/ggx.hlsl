// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_GGX_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf/ggx.hlsl"
#include "nbl/builtin/hlsl/bxdf/base/cook_torrance_base.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class Config>
using SGGXIsotropic = SCookTorrance<Config, ndf::GGX<typename Config::scalar_type, false, ndf::MTT_REFLECT>, fresnel::Conductor<typename Config::spectral_type> >;

template<class Config>
using SGGXAnisotropic = SCookTorrance<Config, ndf::GGX<typename Config::scalar_type, true, ndf::MTT_REFLECT>, fresnel::Conductor<typename Config::spectral_type> >;

}

template<typename C>
struct traits<bxdf::reflection::SGGXIsotropic<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMicrofacet = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename C>
struct traits<bxdf::reflection::SGGXAnisotropic<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMicrofacet = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
