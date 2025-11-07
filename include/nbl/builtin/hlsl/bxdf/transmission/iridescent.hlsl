// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_IRIDESCENT_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRANSMISSION_IRIDESCENT_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/transmission/ggx.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace transmission
{

template<class Config>
using SIridescent = SCookTorrance<Config, ndf::GGX<typename Config::scalar_type, false, ndf::MTT_REFLECT_REFRACT>, fresnel::Iridescent<typename Config::spectral_type, true> >;

}

template<typename C>
struct traits<bxdf::transmission::SIridescent<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMicrofacet = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
