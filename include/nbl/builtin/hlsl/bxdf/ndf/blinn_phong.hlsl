
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_BLINN_PHONG_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_BLINN_PHONG_INCLUDED_

#include <nbl/builtin/hlsl/math/constants.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf.hlsl>
// for beckmann sampling
#include <nbl/builtin/hlsl/bxdf/ndf/beckmann.hlsl>


namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

float blinn_phong(in float NdotH, in float n)
{
    return isinf(n) ? FLT_INF : math::RECIPROCAL_PI*0.5*(n+2.0) * pow(NdotH,n);
}
//ashikhmin-shirley ndf
float blinn_phong(in float NdotH, in float one_minus_NdotH2_rcp, in float TdotH2, in float BdotH2, in float nx, in float ny)
{
    float n = (TdotH2*ny + BdotH2*nx) * one_minus_NdotH2_rcp;

    return (isinf(nx)||isinf(ny)) ?  FLT_INF : sqrt((nx + 2.0)*(ny + 2.0))*math::RECIPROCAL_PI*0.5 * pow(NdotH,n);
}

struct IsotropicBlinnPhong : NDFBase<>
{
	float n;

	//conversion between alpha and Phong exponent, Walter et.al.
	static float phong_exp_to_alpha2(in float _n)
	{
		return 2.0 / (_n + 2.0);
	}
	//+INF for a2==0.0
	static float alpha2_to_phong_exp(in float a2)
	{
		return 2.0 / a2 - 2.0;
	}

	template <class IncomingrayDirInfo>
	static float3 generateH_impl(in surface_interactions::Anisotropic<IncomingrayDirInfo> interaction, inout float3 u, out AnisotropicMicrofacetCache cache, in float _ax, in float _ay)
	{
		IsotropicBeckmann::generateH_impl(interaction, u, cache, _ax, _ay);
	}

	template <class IncomingrayDirInfo>
	float3 generateH(in surface_interactions::Anisotropic<IncomingrayDirInfo> interaction, inout float3 u, out AnisotropicMicrofacetCache cache)
	{
		float a = sqrt( phong_exp_to_alpha2(n) );
		return generateH_impl(interaction, u, cache, a, a);
	}

	float D(in float NdotH2)
	{
		// here doing pow(NdotH2,n*0.5) instead of pow(NdotH,n) to keep compliant to common API (parameter of D() should be NdotH2)
		return isinf(n) ? FLT_INF : math::RECIPROCAL_PI * 0.5 * (n + 2.0) * pow(NdotH2, n*0.5);
	}

	float Lambda(in float NdotX2)
	{
		// TODO
		// eh ill probably just use beckmann's lambda
		return 0.f / 0.f;//nan
	}
};

struct GGX : IsotropicBlinnPhong
{
	//float nx; // inherited from base as `n`
	float ny;

	template <class IncomingrayDirInfo>
	float3 generateH(in surface_interactions::Anisotropic<IncomingrayDirInfo> interaction, inout float3 u, out AnisotropicMicrofacetCache cache)
	{
		float ax = sqrt(phong_exp_to_alpha2(n));
		float ay = sqrt(phong_exp_to_alpha2(ny));
		return generateH_impl(interaction, u, cache, ax, ay);
	}

	float D(in float TdotH2, in float BdotH2, in float NdotH2)
	{
		float aniso_n = (TdotH2 * ny + BdotH2 * n) / (1.0 - NdotH2);

		return (isinf(n) || isinf(ny)) ? FLT_INF : sqrt((n + 2.0) * (ny + 2.0)) * math::RECIPROCAL_PI * 0.5 * pow(NdotH2, aniso_n*0.5);
	}

	float Lambda(in float TdotH2, in float BdotH2, in float NdotX2)
	{
		// TODO
		// eh ill probably just use beckmann's lambda
		return 0.f / 0.f; //nan
	}
};

}
}
}
}

#endif