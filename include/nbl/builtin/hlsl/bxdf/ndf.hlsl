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
    REFLECT_BIT = 0b01,
    REFRACT_BIT = 0b10,
    REFLECT_REFRACT_BIT = 0b11
};

template<typename NDF, uint16_t reflect_refract>
struct microfacet_to_light_measure_transform;


template<typename NDF>
struct microfacet_to_light_measure_transform<NDF,REFLECT_BIT>
{
    using this_t = microfacet_to_light_measure_transform<NDF,REFLECT_BIT>;
    using scalar_type = typename NDF::scalar_type;

    static this_t create(scalar_type NDFcos, scalar_type maxNdotV)
    {
        this_t retval;
        retval.NDFcos = NDFcos;
        if (is_ggx_v<NDF>)
            retval.maxNdotL = maxNdotV;
        else
            retval.maxNdotV = maxNdotV;
        return retval;
    }

    scalar_type operator()()
    {
        if (is_ggx_v<NDF>)
            return NDFcos * maxNdotL;
        else
            return 0.25 * NDFcos / maxNdotV;
    }

    scalar_type NDFcos;
    scalar_type maxNdotV;
    scalar_type maxNdotL;
};

template<typename NDF>
struct microfacet_to_light_measure_transform<NDF,REFRACT_BIT>
{
    using this_t = microfacet_to_light_measure_transform<NDF,REFRACT_BIT>;
    using scalar_type = typename NDF::scalar_type;

    static this_t create(scalar_type NDFcos, scalar_type absNdotV, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        this_t retval;
        retval.NDFcos = NDFcos;
        if (is_ggx_v<NDF>)
            retval.absNdotL = absNdotV;
        else
            retval.absNdotV = absNdotV;
        retval.VdotH = VdotH;
        retval.LdotH = LdotH;
        retval.VdotHLdotH = VdotHLdotH;
        retval.orientedEta = orientedEta;
        return retval;
    }

    scalar_type operator()()
    {
        if (is_ggx_v<NDF>)
        {
            scalar_type denominator = absNdotL;
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

    scalar_type NDFcos;
    scalar_type absNdotV;
    scalar_type absNdotL;

    scalar_type VdotH;
    scalar_type LdotH;
    scalar_type VdotHLdotH;
    scalar_type orientedEta;
};

template<typename NDF>
struct microfacet_to_light_measure_transform<NDF,REFLECT_REFRACT_BIT>
{
    using this_t = microfacet_to_light_measure_transform<NDF,REFLECT_REFRACT_BIT>;
    using scalar_type = typename NDF::scalar_type;

    static this_t create(scalar_type NDFcos, scalar_type absNdotV, bool transmitted, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        this_t retval;
        retval.NDFcos = NDFcos;
        if (is_ggx_v<NDF>)
            retval.absNdotL = absNdotV;
        else
            retval.absNdotV = absNdotV;
        retval.transmitted = transmitted;
        retval.VdotH = VdotH;
        retval.LdotH = LdotH;
        retval.VdotHLdotH = VdotHLdotH;
        retval.orientedEta = orientedEta;
        return retval;
    }

    scalar_type operator()()
    {
        if (is_ggx_v<NDF>)
        {
            scalar_type denominator = absNdotL;
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

    bool transmitted;
    scalar_type NDFcos;
    scalar_type absNdotV;
    scalar_type absNdotL;

    scalar_type VdotH;
    scalar_type LdotH;
    scalar_type VdotHLdotH;
    scalar_type orientedEta;
};

}
}
}
}

#endif