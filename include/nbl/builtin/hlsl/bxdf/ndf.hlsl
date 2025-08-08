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

enum MicrofacetTransformTypes : uint16_t
{
    MTT_REFLECT = 0b01,
    MTT_REFRACT = 0b10,
    MTT_REFLECT_REFRACT = 0b11
};

template<typename T, bool IsGGX, MicrofacetTransformTypes reflect_refract>
struct microfacet_to_light_measure_transform;

template<typename T>
struct microfacet_to_light_measure_transform<T, false, MTT_REFLECT>
{
    using this_t = microfacet_to_light_measure_transform<T, false, MTT_REFLECT>;
    using scalar_type = T;

    // this computes the max(NdotL,0)/(4*max(NdotV,0)*max(NdotL,0)) factor which transforms PDFs in the f in projected microfacet f * NdotH measure to projected light measure f * NdotL
    static scalar_type __call(scalar_type NDFcos, scalar_type maxNdotV)
    {
        return scalar_type(0.25) * NDFcos / maxNdotV;
    }
};

template<typename T>
struct microfacet_to_light_measure_transform<T, true, MTT_REFLECT>
{
    using this_t = microfacet_to_light_measure_transform<T, true, MTT_REFLECT>;
    using scalar_type = T;

    // this computes the max(NdotL,0)/(4*max(NdotV,0)*max(NdotL,0)) factor which transforms PDFs in the f in projected microfacet f * NdotH measure to projected light measure f * NdotL
    static scalar_type __call(scalar_type NDFcos, scalar_type maxNdotL)
    {
        return NDFcos * maxNdotL;
    }
};

template<typename T>
struct microfacet_to_light_measure_transform<T, false, MTT_REFRACT>
{
    using this_t = microfacet_to_light_measure_transform<T, false, MTT_REFRACT>;
    using scalar_type = T;

    static scalar_type __call(scalar_type NDFcos, scalar_type absNdotV, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        scalar_type denominator = absNdotV;
        const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
        // VdotHLdotH is negative under transmission, so thats denominator is negative
        denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
        return NDFcos * VdotHLdotH / denominator;
    }
};

template<typename T>
struct microfacet_to_light_measure_transform<T, true, MTT_REFRACT>
{
    using this_t = microfacet_to_light_measure_transform<T, true, MTT_REFRACT>;
    using scalar_type = T;

    static scalar_type __call(scalar_type NDFcos, scalar_type absNdotV, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        scalar_type denominator = absNdotV;
        const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
        // VdotHLdotH is negative under transmission, so thats denominator is negative
        denominator *= -scalar_type(4.0) * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
        return NDFcos * denominator;
    }
};

template<typename T>
struct microfacet_to_light_measure_transform<T, false, MTT_REFLECT_REFRACT>
{
    using this_t = microfacet_to_light_measure_transform<T, false, MTT_REFLECT_REFRACT>;
    using scalar_type = T;

    static scalar_type __call(scalar_type NDFcos, scalar_type absNdotV, bool transmitted, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        scalar_type denominator = absNdotV;
        if (transmitted)
        {
            const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
            // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
        }
        return NDFcos * (transmitted ? VdotHLdotH : scalar_type(0.25)) / denominator;
    }
};

template<typename T>
struct microfacet_to_light_measure_transform<T, true, MTT_REFLECT_REFRACT>
{
    using this_t = microfacet_to_light_measure_transform<T, true, MTT_REFLECT_REFRACT>;
    using scalar_type = T;

    static scalar_type __call(scalar_type NDFcos, scalar_type absNdotV, bool transmitted, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        scalar_type denominator = absNdotV;
        if (transmitted)
        {
            const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
            // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -scalar_type(4.0) * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
        }
        return NDFcos * denominator;
    }
};

}
}
}
}

#endif