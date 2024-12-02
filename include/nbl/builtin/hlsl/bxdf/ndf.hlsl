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

// common
namespace impl
{
template<typename T, bool ggx = false>
struct microfacet_to_light_measure_transform
{
    static T __call(T NDFcos, T absNdotV, bool transmitted, T VdotH, T LdotH, T VdotHLdotH, T orientedEta)
    {
        T denominator = absNdotV;
        if (transmitted)
        {
            const T VdotH_etaLdotH = (VdotH+orientedEta*LdotH);
            // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
        }
        return NDFcos * (transmitted ? VdotHLdotH : 0.25) / denominator;
    }

    static T __call(T NDFcos, T maxNdotV)
    {
        return 0.25 * NDFcos / maxNdotV;
    }

};

template<typename T>
struct microfacet_to_light_measure_transform<T,true>
{
    static T __call(T NDFcos_already_in_reflective_dL_measure, T absNdotL, bool transmitted, T VdotH, T LdotH, T VdotHLdotH, T orientedEta)
    {
        T denominator = absNdotL;
        if (transmitted)
        {
            const T VdotH_etaLdotH = (VdotH+orientedEta*LdotH);
            // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
        }
        return NDFcos_already_in_reflective_dL_measure * (transmitted ? VdotHLdotH : 0.25) / denominator;
    }

    static T __call(T NDFcos_already_in_reflective_dL_measure, T maxNdotL)
    {
        return NDFcos_already_in_reflective_dL_measure * maxNdotL;
    }
};
}

template<typename T, bool ggx NBL_FUNC_REQUIRES(is_scalar_v<T>)
T microfacet_to_light_measure_transform(T NDFcos, T absNdotV, bool transmitted, T VdotH, T LdotH, T VdotHLdotH, T orientedEta)
{
    return impl::microfacet_to_light_measure_transform<T,ggx>::__call(NDFcos, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
}

template<typename T, bool ggx NBL_FUNC_REQUIRES(is_scalar_v<T>)
T microfacet_to_light_measure_transform(T NDFcos, T maxNdotV)
{
    return impl::microfacet_to_light_measure_transform<T,ggx>::__call(NDFcos, maxNdotV);
}


// blinn-phong
template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T blinn_phong(T NdotH, T n)
{
    return isinf(n) ? numeric_limits<T>::infinity : numbers::inv_pi<T> * 0.5 * (n + 2.0) * pow(NdotH,n);
}
//ashikhmin-shirley ndf
template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T blinn_phong(T NdotH, T one_minus_NdotH2_rcp, T TdotH2, T BdotH2, T nx, T ny)
{
    T n = (TdotH2 * ny + BdotH2 * nx) * one_minus_NdotH2_rcp;
    return (isinf(nx) || isinf(ny)) ?  numeric_limits<T>::infinity : sqrt((nx + 2.0) * (ny + 2.0)) * numbers::inv_pi<T> * 0.5 * pow(NdotH, n);
}


// beckmann
template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann(T a2, T NdotH2)
{
    T nom = exp( (NdotH2 - 1.0) / (a2 * NdotH2) ); // exp(x) == exp2(x/log(2)) ?
    T denom = a2 * NdotH2 * NdotH2;
    return numbers::inv_pi<T> * nom / denom;
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T beckmann(T ax, T ay, T ax2, T ay2, T TdotH2, T BdotH2, T NdotH2)
{
    T nom = exp(-(TdotH2 / ax2 + BdotH2 / ay2) / NdotH2);
    T denom = ax * ay * NdotH2 * NdotH2;
    return numbers::inv_pi<T> * nom / denom;
}


// ggx
template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_trowbridge_reitz(T a2, T NdotH2)
{
    T denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2* numbers::inv_pi<T> / (denom * denom);
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_burley_aniso(T anisotropy, T a2, T TdotH, T BdotH, T NdotH)
{
	T antiAniso = 1.0 - anisotropy;
	T atab = a2 * antiAniso;
	T anisoTdotH = antiAniso * TdotH;
	T anisoNdotH = antiAniso * NdotH;
	T w2 = antiAniso/(BdotH * BdotH + anisoTdotH * anisoTdotH + anisoNdotH * anisoNdotH * a2);
	return w2 * w2 * atab * numbers::inv_pi<T>;
}

template<typename T NBL_FUNC_REQUIRES(is_scalar_v<T>)
T ggx_aniso(T TdotH2, T BdotH2, T NdotH2, T ax, T ay, T ax2, T ay2)
{
	T a2 = ax * ay;
	T denom = TdotH2 / ax2 + BdotH2 / ay2 + NdotH2;
	return numbers::inv_pi<T> / (a2 * denom * denom);
}

}
}
}
}

#endif