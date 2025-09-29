// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_REFLECTION_LAMBERTIAN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_REFLECTION_LAMBERTIAN_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/bxdf_traits.hlsl"
#include "nbl/builtin/hlsl/bxdf/base/lambertian.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace reflection
{

template<class Config>
using SLambertian = base::SLambertianBase<Config, false>;

}

template<typename C>
struct traits<bxdf::reflection::SLambertian<C> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
