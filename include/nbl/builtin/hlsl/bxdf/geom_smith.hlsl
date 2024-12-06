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

// TODO: need struct specializations? don't know which is used vs. helper

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T G1(T lambda)
{
    return 1.0 / (1.0 + lambda);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T VNDF_pdf_wo_clamps(T ndf, T lambda_V, T maxNdotV, out T onePlusLambda_V)
{
    onePlusLambda_V = 1.0 + lambda_V;

    return ndf::microfacet_to_light_measure_transform<T>(ndf / onePlusLambda_V, maxNdotV);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T VNDF_pdf_wo_clamps(T ndf, T lambda_V, T absNdotV, bool transmitted, T VdotH, T LdotH, T VdotHLdotH, T orientedEta, T reflectance, out T onePlusLambda_V)
{
    onePlusLambda_V = 1.0 + lambda_V;

    return ndf::microfacet_to_light_measure_transform<T>((transmitted ? (1.0 - reflectance) : reflectance) * ndf / onePlusLambda_V, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
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


// beckmann
template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann_C2(T NdotX2, T a2)
{
    return NdotX2 / (a2 * (1.0 - NdotX2));
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann_C2(T TdotX2, T BdotX2, T NdotX2, T ax2, T ay2)
{
    return NdotX2/(TdotX2 * ax2 + BdotX2 * ay2);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann_Lambda(T c2)
{
    T c = sqrt(c2);
    T nom = 1.0 - 1.259 * c + 0.396 * c2;
    T denom = 2.181 * c2 + 3.535 * c;
    return lerp(0.0, nom / denom, c < 1.6);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann_Lambda(T NdotX2, T a2)
{
    return beckmann_Lambda<T>(beckmann_C2<T>(NdotX2, a2));
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann_Lambda(T TdotX2, T BdotX2, T NdotX2, T ax2, T ay2)
{
    return beckmann_Lambda<T>(beckmann_C2<T>(TdotX2, BdotX2, NdotX2, ax2, ay2));
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann_smith_correlated(T NdotV2, T NdotL2, T a2)
{
    T c2 = beckmann_C2<T>(NdotV2, a2);
    T L_v = beckmann_Lambda<T>(c2);
    c2 = beckmann_C2<T>(NdotL2, a2);
    T L_l = beckmann_Lambda<T>(c2);
    return G1<T>(L_v + L_l);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann_smith_correlated(T TdotV2, T BdotV2, T NdotV2, T TdotL2, T BdotL2, T NdotL2, T ax2, T ay2)
{
    T c2 = beckmann_C2<T>(TdotV2, BdotV2, NdotV2, ax2, ay2);
    T L_v = beckmann_Lambda<T>(c2);
    c2 = beckmann_C2<T>(TdotL2, BdotL2, NdotL2, ax2, ay2);
    T L_l = beckmann_Lambda<T>(c2);
    return G1<T>(L_v + L_l);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann_smith_G2_over_G1(T lambdaV_plus_one, T NdotL2, T a2)
{
    T lambdaL = beckmann_Lambda<T>(NdotL2, a2);
    return lambdaV_plus_one / (lambdaV_plus_one+lambdaL);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann_smith_G2_over_G1(T lambdaV_plus_one, T TdotL2, T BdotL2, T NdotL2, T ax2, T ay2)
{
    T c2 = beckmann_C2<T>(TdotL2, BdotL2, NdotL2, ax2, ay2);
	T lambdaL = beckmann_Lambda<T>(c2);
    return lambdaV_plus_one / (lambdaV_plus_one + lambdaL);
}


// ggx
template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_devsh_part(T NdotX2, T a2, T one_minus_a2)
{
    return sqrt(a2 + one_minus_a2 * NdotX2);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_devsh_part(T TdotX2, T BdotX2, T NdotX2, T ax2, T ay2)
{
    return sqrt(TdotX2 * ax2 + BdotX2 * ay2 + NdotX2);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_G1_wo_numerator(T NdotX, T NdotX2, T a2, T one_minus_a2)
{
    return 1.0 / (NdotX + ggx_devsh_part<T>(NdotX2,a2,one_minus_a2));
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_G1_wo_numerator(T NdotX, T TdotX2, T BdotX2, T NdotX2, T ax2, T ay2)
{
    return 1.0 / (NdotX + ggx_devsh_part<T>(TdotX2, BdotX2, NdotX2, ax2, ay2));
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_G1_wo_numerator(T NdotX, T devsh_part)
{
    return 1.0 / (NdotX + devsh_part);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_correlated_wo_numerator(T NdotV, T NdotV2, T NdotL, T NdotL2, T a2, T one_minus_a2)
{
    T Vterm = NdotL * ggx_devsh_part<T>(NdotV2,a2,one_minus_a2);
    T Lterm = NdotV * ggx_devsh_part<T>(NdotL2,a2,one_minus_a2);
    return 0.5 / (Vterm + Lterm);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_correlated_wo_numerator(T NdotV, T NdotV2, T NdotL, T NdotL2, T a2)
{
    return ggx_correlated_wo_numerator<T>(NdotV,NdotV2,NdotL,NdotL2,a2,1.0 - a2);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_correlated_wo_numerator(T NdotV, T TdotV2, T BdotV2, T NdotV2, T NdotL, T TdotL2, T BdotL2, T NdotL2, T ax2, T ay2)
{
    T Vterm = NdotL * ggx_devsh_part<T>(TdotV2,BdotV2,NdotV2,ax2,ay2);
    T Lterm = NdotV * ggx_devsh_part<T>(TdotL2,BdotL2,NdotL2,ax2,ay2);
    return 0.5 / (Vterm + Lterm);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_G2_over_G1(T NdotL, T NdotL2, T NdotV, T NdotV2, T a2, T one_minus_a2)
{
    T devsh_v = ggx_devsh_part<T>(NdotV2,a2,one_minus_a2);
	T G2_over_G1 = NdotL*(devsh_v + NdotV); // alternative `Vterm+NdotL*NdotV /// NdotL*NdotV could come as a parameter
	G2_over_G1 /= NdotV*ggx_devsh_part<T>(NdotL2,a2,one_minus_a2) + NdotL*devsh_v;

    return G2_over_G1;
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_G2_over_G1_devsh(T NdotL, T NdotL2, T NdotV, T devsh_v, T a2, T one_minus_a2)
{
	T G2_over_G1 = NdotL*(devsh_v + NdotV); // alternative `Vterm+NdotL*NdotV /// NdotL*NdotV could come as a parameter
	G2_over_G1 /= NdotV*ggx_devsh_part<T>(NdotL2,a2,one_minus_a2) + NdotL*devsh_v;

    return G2_over_G1;
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_G2_over_G1(T NdotL, T TdotL2, T BdotL2, T NdotL2, T NdotV, T TdotV2, T BdotV2, T NdotV2, T ax2, T ay2)
{
    T devsh_v = ggx_devsh_part<T>(TdotV2,BdotV2,NdotV2,ax2,ay2);
	T G2_over_G1 = NdotL*(devsh_v + NdotV);
	G2_over_G1 /= NdotV*ggx_devsh_part<T>(TdotL2,BdotL2,NdotL2,ax2,ay2) + NdotL*devsh_v;

    return G2_over_G1;
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_G2_over_G1_devsh(T NdotL, T TdotL2, T BdotL2, T NdotL2, T NdotV, T devsh_v, T ax2, T ay2)
{
	T G2_over_G1 = NdotL*(devsh_v + NdotV);
	G2_over_G1 /= NdotV*ggx_devsh_part<T>(TdotL2,BdotL2,NdotL2,ax2,ay2) + NdotL*devsh_v;

    return G2_over_G1;
}

}
}
}
}

#endif