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

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct SIsotropicParams
{
    using this_t = SIsotropicParams<T>;

    static this_t create(T NdotH, T n)  // blinn-phong
    {
        this_t retval;
        retval.NdotH = NdotH;
        retval.n = n;
        return this_t;
    }

    static this_t create(T a2, T NdotH2)    // beckmann, ggx
    {
        this_t retval;
        retval.a2 = a2;
        retval.NdotH = NdotH;
        return this_t;
    }

    T a2;
    T n;
    T NdotH;
    T NdotH2;
    T TdotH2;
    T BdotH2;
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct SAnisotropicParams
{
    using this_t = SAnisotropicParams<T>;

    static this_t create(T NdotH, T one_minus_NdotH2_rcp, T TdotH2, T BdotH2, T nx, T ny)   // blinn-phong
    {
        this_t retval;
        retval.NdotH = NdotH;
        retval.one_minus_NdotH2_rcp = one_minus_NdotH2_rcp;
        retval.TdotH2 = TdotH2;
        retval.BdotH2 = BdotH2;
        retval.nx = nx;
        retval.ny = ny;
        return this_t;
    }

    static this_t create(T ax, T ay, T ax2, T ay2, T TdotH2, T BdotH2, T NdotH2)    // beckmann, ggx aniso
    {
        this_t retval;
        retval.ax = ax;
        retval.ax2 = ax2;
        retval.ay = ay;
        retval.ay2 = ay2;
        retval.TdotH2 = TdotH2;
        retval.BdotH2 = BdotH2;
        retval.NdotH2 = NdotH2;
        return this_t;
    }

    static this_t create(T a2, T TdotH, T BdotH, T NdotH)   // ggx burley
    {
        this_t retval;
        retval.ax = a2;
        retval.TdotH = TdotH;
        retval.BdotH = BdotH;
        retval.NdotH = NdotH;
        return this_t;
    }

    T ax;
    T ay;
    T ax2;
    T ay2;
    T nx;
    T ny;
    T NdotH;
    T TdotH;
    T BdotH;
    T NdotH2;
    T TdotH2;
    T BdotH2;
    T one_minus_NdotH2_rcp;
};


template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct BlinnPhong
{
    using scalar_type = T;

    // blinn-phong
    scalar_type operator()(SIsotropicParams<scalar_type> params)
    {
        // n is shininess exponent in original paper
        return isinf<scalar_type>(params.n) ? numeric_limits<scalar_type>::infinity : numbers::inv_pi<scalar_type> * 0.5 * (params.n + 2.0) * pow<scalar_type>(params.NdotH, params.n);
    }

    //ashikhmin-shirley ndf
    scalar_type operator()(SAnisotropicParams<scalar_type> params)
    {
        scalar_type n = (params.TdotH2 * params.ny + params.BdotH2 * params.nx) * params.one_minus_NdotH2_rcp;
        return (isinf<scalar_type>(params.nx) || isinf<scalar_type>(params.ny)) ?  numeric_limits<scalar_type>::infinity : 
            sqrt<scalar_type>((params.nx + 2.0) * (params.ny + 2.0)) * numbers::inv_pi<scalar_type> * 0.5 * pow<scalar_type>(params.NdotH, n);
    }
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct Beckmann
{
    using scalar_type = T;

    scalar_type operator()(SIsotropicParams<scalar_type> params)
    {
        scalar_type nom = exp<scalar_type>( (params.NdotH2 - 1.0) / (params.a2 * params.NdotH2) ); // exp(x) == exp2(x/log(2)) ?
        scalar_type denom = params.a2 * params.NdotH2 * params.NdotH2;
        return numbers::inv_pi<scalar_type> * nom / denom;
    }

    scalar_type operator()(SAnisotropicParams<scalar_type> params)
    {
        scalar_type nom = exp<scalar_type>(-(params.TdotH2 / params.ax2 + params.BdotH2 / params.ay2) / params.NdotH2);
        scalar_type denom = params.ax * params.ay * params.NdotH2 * params.NdotH2;
        return numbers::inv_pi<scalar_type> * nom / denom;
    }
};


template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct GGX
{
    using scalar_type = T;

    // trowbridge-reitz
    scalar_type operator()(SIsotropicParams<scalar_type> params)
    {
        scalar_type denom = params.NdotH2 * (params.a2 - 1.0) + 1.0;
        return params.a2 * numbers::inv_pi<scalar_type> / (denom * denom);
    }

    scalar_type operator()(SAnisotropicParams<scalar_type> params)
    {
        scalar_type a2 = params.ax * params.ay;
        scalar_type denom = params.TdotH2 / params.ax2 + params.BdotH2 / params.ay2 + params.NdotH2;
        return numbers::inv_pi<scalar_type> / (params.a2 * denom * denom);
    }

    // burley
    scalar_type operator()(SAnisotropicParams<scalar_type> params, scalar_type anisotropy)
    {
        scalar_type antiAniso = 1.0 - anisotropy;
        scalar_type atab = params.ax * antiAniso;
        scalar_type anisoTdotH = antiAniso * params.TdotH;
        scalar_type anisoNdotH = antiAniso * params.NdotH;
        scalar_type w2 = antiAniso/(params.BdotH * params.BdotH + anisoTdotH * anisoTdotH + anisoNdotH * anisoNdotH * params.ax);
        return w2 * w2 * atab * numbers::inv_pi<scalar_type>;
    }
};

// common
namespace impl
{
template<class T, class U>
struct is_ggx : bool_constant<
    is_same<T, GGX<U> >::value
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
        if (is_ggv_v<NDF>)
            retval.maxNdotL = maxNdotV;
        else
            retval.maxNdotV = maxNdotV;
        return retval;
    }

    scalar_type operator()()
    {
        if (is_ggv_v<NDF>)
            return NDFcos * maxNdotL;
        else
            return 0.25 * NDFcos / maxNdotV;
    }

    scalar_type NDFcos
    scalar_type maxNdotV
    scalar_type maxNdotL
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
        if (is_ggv_v<NDF>)
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
        scalar_type denominator;
        if (is_ggv_v<NDF>)
            denominator = absNdotL;
        else
            denominator = absNdotV;

        const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
        // VdotHLdotH is negative under transmission, so thats denominator is negative
        denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
        return NDFcos * VdotHLdotH / denominator;
    }

    scalar_type NDFcos
    scalar_type absNdotV
    scalar_type absNdotL

    scalar_type VdotH
    scalar_type LdotH
    scalar_type VdotHLdotH
    scalar_type orientedEta
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
        if (is_ggv_v<NDF>)
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
        if (is_ggv_v<NDF>)
        {
            scalar_type denominator = absNdotL;
            if (transmitted)
            {
                const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
                // VdotHLdotH is negative under transmission, so thats denominator is negative
                denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
            }
            return NDFcos * (transmitted ? VdotHLdotH : 0.25) / denominator;
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

    bool transmitted
    scalar_type NDFcos
    scalar_type absNdotV
    scalar_type absNdotL

    scalar_type VdotH
    scalar_type LdotH
    scalar_type VdotHLdotH
    scalar_type orientedEta
};

}
}
}
}

#endif