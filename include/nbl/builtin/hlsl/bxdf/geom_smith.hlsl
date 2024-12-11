// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_GEOM_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_GEOM_INCLUDED_

#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace smith
{


template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T G1(T lambda)
{
    return 1.0 / (1.0 + lambda);
}

template<typename NDF>
T VNDF_pdf_wo_clamps(NDF ndf, typename NDF::scalar_type lambda_V, typename NDF::scalar_type maxNdotV, out typename NDF::scalar_type onePlusLambda_V)
{
    onePlusLambda_V = 1.0 + lambda_V;
    ndf::microfacet_to_light_measure_transform<NDF,ndf::REFLECT_BIT> transform = ndf::microfacet_to_light_measure_transform<NDF,ndf::REFLECT_BIT>::create(ndf() / onePlusLambda_V, maxNdotV);
    return transform();
}

template<typename NDF>
T VNDF_pdf_wo_clamps(NDF ndf, typename NDF::scalar_type lambda_V, typename NDF::scalar_type absNdotV, bool transmitted, typename NDF::scalar_type VdotH, typename NDF::scalar_type LdotH, typename NDF::scalar_type VdotHLdotH, typename NDF::scalar_type orientedEta, typename NDF::scalar_type reflectance, out typename NDF::scalar_type onePlusLambda_V)
{
    onePlusLambda_V = 1.0 + lambda_V;
    ndf::microfacet_to_light_measure_transform<NDF,ndf::REFLECT_REFRACT_BIT> transform 
        = ndf::microfacet_to_light_measure_transform<NDF,ndf::REFLECT_REFRACT_BIT>::create((transmitted ? (1.0 - reflectance) : reflectance) * ndf() / onePlusLambda_V, , absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
    return transform();
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T VNDF_pdf_wo_clamps(T ndf, T G1_over_2NdotV)
{
    return ndf * 0.5 * G1_over_2NdotV;
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T FVNDF_pdf_wo_clamps(T fresnel_ndf, T G1_over_2NdotV, T absNdotV, bool transmitted, T VdotH, T LdotH, T VdotHLdotH, T orientedEta)
{
    T FNG = fresnel_ndf * G1_over_2NdotV;
    T factor = 0.5;
    if (transmitted)
    {
        const T VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
        // VdotHLdotH is negative under transmission, so this factor is negative
        factor *= -2.0 * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
    }
    return FNG * factor;
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T VNDF_pdf_wo_clamps(T ndf, T G1_over_2NdotV, T absNdotV, bool transmitted, T VdotH, T LdotH, T VdotHLdotH, T orientedEta, T reflectance)
{
    T FN = (transmitted ? (1.0 - reflectance) : reflectance) * ndf;
    return FVNDF_pdf_wo_clamps<T>(FN, G1_over_2NdotV, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
}


template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct SIsotropicParams
{
    using this_t = SIsotropicParams<T>;

    static this_t create(T a2, T NdotV2, T NdotL2, T lambdaV_plus_one)  // beckmann
    {
        this_t retval;
        retval.a2 = a2;
        retval.NdotV2 = NdotV2;
        retval.NdotL2 = NdotL2;
        retval.lambdaV_plus_one = lambdaV_plus_one;
        return this_t;
    }

    static this_t create(T a2, T NdotV, T NdotV2, T NdotL, T NdotL2)    // ggx
    {
        this_t retval;
        retval.a2 = a2;
        retval.NdotV = NdotV;
        retval.NdotV2 = NdotV2;
        retval.NdotL = NdotL;
        retval.NdotL2 = NdotL2;
        retval.one_minus_a2 = 1.0 - a2;
        return this_t;
    }

    T a2;
    T NdotV;
    T NdotL;
    T NdotV2;
    T NdotL2;
    T lambdaV_plus_one;
    T one_minus_a2;
};

template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct SAnisotropicParams
{
    using this_t = SAnisotropicParams<T>;

    static this_t create(T ax2, T ay2, T TdotV2, T BdotV2, T NdotV2, T TdotL2, T BdotL2, T NdotL2, T lambdaV_plus_one)  // beckmann
    {
        this_t retval;
        retval.ax2 = ax2;
        retval.ay2 = ay2;
        retval.TdotV2 = TdotV2;
        retval.BdotV2 = BdotV2;
        retval.NdotV2 = NdotV2;
        retval.TdotL2 = TdotL2;
        retval.BdotL2 = BdotL2;
        retval.NdotL2 = NdotL2;
        retval.lambdaV_plus_one = lambdaV_plus_one;
        return this_t;
    }

    static this_t create(T ax2, T ay2, T NdotV, T TdotV2, T BdotV2, T NdotV2, T NdotL, T TdotL2, T BdotL2, T NdotL2)    // ggx
    {
        this_t retval;
        retval.ax2 = ax2;
        retval.ay2 = ay2;
        retval.NdotL = NdotL;
        retval.NdotV = NdotV;
        retval.TdotV2 = TdotV2;
        retval.BdotV2 = BdotV2;
        retval.NdotV2 = NdotV2;
        retval.TdotL2 = TdotL2;
        retval.BdotL2 = BdotL2;
        retval.NdotL2 = NdotL2;
        return this_t;
    }

    T ax2;
    T ay2;
    T NdotV;
    T NdotL;
    T TdotV2;
    T BdotV2;
    T NdotV2;
    T TdotL2;
    T BdotL2;
    T NdotL2;
    T lambdaV_plus_one;
};


// beckmann
template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct Beckmann
{
    using scalar_type = T;

    scalar_type C2(scalar_type NdotX2, scalar_type a2)
    {
        return NdotX2 / (a2 * (1.0 - NdotX2));    
    }

    scalar_type C2(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2, scalar_type ax2, scalar_type ay2)
    {
        return NdotX2 / (TdotX2 * ax2 + BdotX2 * ay2);
    }

    scalar_type Lambda(scalar_type c2)
    {
        scalar_type c = sqrt(c2);
        scalar_type nom = 1.0 - 1.259 * c + 0.396 * c2;
        scalar_type denom = 2.181 * c2 + 3.535 * c;
        return lerp(0.0, nom / denom, c < 1.6);
    }

    scalar_type Lambda(scalar_type NdotX2, scalar_type a2)
    {
        return Lambda(C2(NdotX2, a2));
    }

    scalar_type Lambda(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2, scalar_type ax2, scalar_type ay2)
    {
        return Lambda(C2(TdotX2, BdotX2, NdotX2, ax2, ay2));
    }

    scalar_type smith_correlated(SIsotropicParams<scalar_type> params)
    {
        scalar_type c2 = C2(params.NdotV2, params.a2);
        scalar_type L_v = Lambda(c2);
        c2 = C2(params.NdotL2, params.a2);
        scalar_type L_l = Lambda(c2);
        return G1<scalar_type>(L_v + L_l);
    }

    scalar_type smith_correlated(SAnisotropicParams<scalar_type> params)
    {
        scalar_type c2 = C2(params.TdotV2, params.BdotV2, params.NdotV2, params.ax2, params.ay2);
        scalar_type L_v = Lambda(c2);
        c2 = C2(params.TdotL2, params.BdotL2, params.NdotL2, params.ax2, params.ay2);
        scalar_type L_l = Lambda(c2);
        return G1<scalar_type>(L_v + L_l);
    }

    scalar_type smith_G2_over_G1(SIsotropicParams<scalar_type> params)
    {
        scalar_type lambdaL = Lambda(params.NdotL2, params.a2);
        return params.lambdaV_plus_one / (params.lambdaV_plus_one + lambdaL);
    }

    scalar_type smith_G2_over_G1(SAnisotropicParams<scalar_type> params)
    {
        scalar_type c2 = C2(params.TdotL2, params.BdotL2, params.NdotL2, params.ax2, params.ay2);
        scalar_type lambdaL = Lambda(c2);
        return params.lambdaV_plus_one / (params.lambdaV_plus_one + lambdaL);
    }
};


// ggx
template<typename T NBL_PRIMARY_REQUIRES(is_scalar_v<T>)
struct GGX
{
    using scalar_type = T;

    scalar_type devsh_part(scalar_type NdotX2, scalar_type a2, scalar_type one_minus_a2)
    {
        return sqrt(a2 + one_minus_a2 * NdotX2);
    }

    scalar_type devsh_part(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2, scalar_type ax2, scalar_type ay2)
    {
        return sqrt(TdotX2 * ax2 + BdotX2 * ay2 + NdotX2);
    }

    scalar_type G1_wo_numerator(scalar_type NdotX, scalar_type NdotX2, scalar_type a2, scalar_type one_minus_a2)
    {
        return 1.0 / (NdotX + ggx_devsh_part<T>(NdotX2,a2,one_minus_a2));
    }

    scalar_type G1_wo_numerator(scalar_type NdotX, scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2, scalar_type ax2, scalar_type ay2)
    {
        return 1.0 / (NdotX + ggx_devsh_part<T>(TdotX2, BdotX2, NdotX2, ax2, ay2));
    }

    scalar_type G1_wo_numerator(scalar_type NdotX, scalar_type devsh_part)
    {
        return 1.0 / (NdotX + devsh_part);
    }

    scalar_type correlated_wo_numerator(SIsotropicParams<scalar_type> params)
    {
        scalar_type Vterm = params.NdotL * devsh_part(params.NdotV2, params.a2, params.one_minus_a2);
        scalar_type Lterm = params.NdotV * devsh_part(params.NdotL2, params.a2, params.one_minus_a2);
        return 0.5 / (Vterm + Lterm);
    }

    scalar_type correlated_wo_numerator(SAnisotropicParams<scalar_type> params)
    {
        scalar_type Vterm = params.NdotL * devsh_part(params.TdotV2, params.BdotV2, params.NdotV2, params.ax2, params.ay2);
        scalar_type Lterm = params.NdotV * devsh_part(params.TdotL2, params.BdotL2, params.NdotL2, params.ax2, params.ay2);
        return 0.5 / (Vterm + Lterm);
    }

    scalar_type G2_over_G1(SIsotropicParams<scalar_type> params)
    {
        scalar_type devsh_v = devsh_part(params.NdotV2, params.a2, params.one_minus_a2);
        scalar_type G2_over_G1 = params.NdotL * (devsh_v + params.NdotV); // alternative `Vterm+NdotL*NdotV /// NdotL*NdotV could come as a parameter
        G2_over_G1 /= params.NdotV * devsh_part(params.NdotL2, params.a2, params.one_minus_a2) + params.NdotL * devsh_v;

        return G2_over_G1;
    }

    scalar_type G2_over_G1(SAnisotropicParams<scalar_type> params)
    {
        scalar_type devsh_v = devsh_part(params.TdotV2, params.BdotV2, params.NdotV2, params.ax2, params.ay2);
        scalar_type G2_over_G1 = params.NdotL * (devsh_v + params.NdotV);
        G2_over_G1 /= params.NdotV * devsh_part(params.TdotL2, params.BdotL2, params.NdotL2, params.ax2, params.ay2) + params.NdotL * devsh_v;

        return G2_over_G1;
    }

};

}
}
}
}

#endif