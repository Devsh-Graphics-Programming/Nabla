// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_TRAITS_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/common.hlsl"
#include "nbl/builtin/hlsl/bxdf/reflection.hlsl"
#include "nbl/builtin/hlsl/bxdf/transmission.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{

enum BxDFType : uint16_t
{
    BT_BSDF = 0,
    BT_BRDF,
    BT_BTDF
};

template<typename BxdfT>
struct traits;

template<typename L, typename I, typename A, typename S>
struct traits<bxdf::reflection::SLambertianBxDF<L, I, A, S> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename L, typename I, typename A, typename S>
struct traits<bxdf::reflection::SOrenNayarBxDF<L, I, A, S> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

// no blinn phong

template<typename L, typename I, typename A, typename S>
struct traits<bxdf::reflection::SBeckmannBxDF<L, I, A, S> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename L, typename I, typename A, typename S>
struct traits<bxdf::reflection::SGGXBxDF<L, I, A, S> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BRDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};


template<typename L, typename I, typename A, typename S>
struct traits<bxdf::transmission::SLambertianBxDF<L, I, A, S> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = false;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename L, typename I, typename A, typename S>
struct traits<bxdf::transmission::SSmoothDielectricBxDF<L, I, A, S, false> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename L, typename I, typename A, typename S>
struct traits<bxdf::transmission::SSmoothDielectricBxDF<L, I, A, S, true> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename L, typename I, typename A, typename S>
struct traits<bxdf::transmission::SBeckmannDielectricBxDF<L, I, A, S> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

template<typename L, typename I, typename A, typename S>
struct traits<bxdf::transmission::SGGXDielectricBxDF<L, I, A, S> >
{
    NBL_CONSTEXPR_STATIC_INLINE BxDFType type = BT_BSDF;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotV = true;
    NBL_CONSTEXPR_STATIC_INLINE bool clampNdotL = true;
};

}
}
}

#endif
