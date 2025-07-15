// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

template<typename T, bool IsAnisotropic=false>
struct Beckmann;

template<typename T, bool IsAnisotropic=false>
struct GGX;

// common
namespace impl
{
template<class T, class U>
struct is_ggx : bool_constant<
    is_same<T, GGX<U,false> >::value ||
    is_same<T, GGX<U,true> >::value
> {};
}

template<class T> 
struct is_ggx : impl::is_ggx<T, typename T::scalar_type> {};

template<typename T>
NBL_CONSTEXPR bool is_ggx_v = is_ggx<T>::value;


enum MicrofacetTransformTypes : uint16_t
{
    MTT_REFLECT = 0b01,
    MTT_REFRACT = 0b10,
    MTT_REFLECT_REFRACT = 0b11
};

template<typename NDF, MicrofacetTransformTypes reflect_refract>
struct microfacet_to_light_measure_transform;

template<typename NDF>
struct microfacet_to_light_measure_transform<NDF,MTT_REFLECT>
{
    using this_t = microfacet_to_light_measure_transform<NDF,MTT_REFLECT>;
    using scalar_type = typename NDF::scalar_type;

    // this computes the max(NdotL,0)/(4*max(NdotV,0)*max(NdotL,0)) factor which transforms PDFs in the f in projected microfacet f * NdotH measure to projected light measure f * NdotL
    static scalar_type __call(scalar_type NDFcos, scalar_type maxNdotV /* or maxNdotL for GGX*/)
    {
        if (is_ggx_v<NDF>)
            return NDFcos * maxNdotV;
        else
            return 0.25 * NDFcos / maxNdotV;
    }
};

template<typename NDF>
struct microfacet_to_light_measure_transform<NDF,MTT_REFRACT>
{
    using this_t = microfacet_to_light_measure_transform<NDF,MTT_REFRACT>;
    using scalar_type = typename NDF::scalar_type;

    static scalar_type __call(scalar_type NDFcos, scalar_type absNdotV, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        if (is_ggx_v<NDF>)
        {
            scalar_type denominator = absNdotV;
            const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
            // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -4.0 * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
            return NDFcos * denominator;
        }
        else
        {
            scalar_type denominator = absNdotV;
            const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
            // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
            return NDFcos * VdotHLdotH / denominator;
        }
    }
};

template<typename NDF>
struct microfacet_to_light_measure_transform<NDF,MTT_REFLECT_REFRACT>
{
    using this_t = microfacet_to_light_measure_transform<NDF,MTT_REFLECT_REFRACT>;
    using scalar_type = typename NDF::scalar_type;

    static scalar_type __call(scalar_type NDFcos, scalar_type absNdotV, bool transmitted, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        if (is_ggx_v<NDF>)
        {
            scalar_type denominator = absNdotV;
            if (transmitted)
            {
                const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
                // VdotHLdotH is negative under transmission, so thats denominator is negative
                denominator *= -4.0 * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
            }
            return NDFcos * denominator;
        }
        else
        {
            scalar_type denominator = absNdotV;
            if (transmitted)
            {
                const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
                // VdotHLdotH is negative under transmission, so thats denominator is negative
                denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
            }
            return NDFcos * (transmitted ? VdotHLdotH : 0.25) / denominator;
        }
    }
};

}
}
}
}

#endif