
// Copyright (C) 2018-2022 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_

#include <nbl/builtin/hlsl/math/constants.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/bxdf/common.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf.hlsl>


namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{
namespace ggx
{


float trowbridge_reitz(in float a2, in float NdotH2)
{
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2* math::RECIPROCAL_PI / (denom*denom);
}

float burley_aniso(float anisotropy, float a2, float TdotH, float BdotH, float NdotH)
{
	float antiAniso = 1.0-anisotropy;
	float atab = a2*antiAniso;
	float anisoTdotH = antiAniso*TdotH;
	float anisoNdotH = antiAniso*NdotH;
	float w2 = antiAniso/(BdotH*BdotH+anisoTdotH*anisoTdotH+anisoNdotH*anisoNdotH*a2);
	return w2*w2*atab * math::RECIPROCAL_PI;
}

float aniso(in float TdotH2, in float BdotH2, in float NdotH2, in float ax, in float ay, in float ax2, in float ay2)
{
	float a2 = ax*ay;
	float denom = TdotH2/ax2 + BdotH2/ay2 + NdotH2;
	return math::RECIPROCAL_PI / (a2 * denom * denom);
}
	

struct IsotropicGGX : NDFBase<>
{
	float a;
	float a2;

	static IsotropicGGX create(float _a)
	{
		IsotropicGGX b;
		b.a = _a;
		b.a2 = _a * _a;
		return b;
	}
	static IsotropicGGX create(float _a, float _a2)
	{
		IsotropicGGX b;
		b.a = _a;
		b.a2 = _a2;
		return b;
	}

	template <class IncomingrayDirInfo>
	static float3 generateH_impl(in surface_interactions::Anisotropic<IncomingrayDirInfo> interaction, inout float3 u, out AnisotropicMicrofacetCache cache, in float _ax, in float _ay)
	{
		const float3 localV = interaction.getTangentSpaceV();

		float3 V = normalize(float3(_ax * localV.x, _ay * localV.y, localV.z));//stretch view vector so that we're sampling as if roughness=1.0

		float lensq = V.x * V.x + V.y * V.y;
		float3 T1 = lensq > 0.0 ? float3(-V.y, V.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
		float3 T2 = cross(V, T1);

		float r = sqrt(u.x);
		float phi = 2.0 * math::PI * u.y;
		float t1 = r * cos(phi);
		float t2 = r * sin(phi);
		float s = 0.5 * (1.0 + V.z);
		t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

		//reprojection onto hemisphere
		//TODO try it wothout the& max(), not sure if -t1*t1-t2*t2>-1.0
		float3 H = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;
		//unstretch
		const float3 localH = normalize(float3(_ax * H.x, _ay * H.y, H.z));

		return localH;
	}

	template <class IncomingrayDirInfo>
	float3 generateH(in surface_interactions::Anisotropic<IncomingrayDirInfo> interaction, inout float3 u, out AnisotropicMicrofacetCache cache)
	{
		return generateH_impl(interaction, u, cache, a, a);
	}

	float D(in float NdotH2)
	{
		float denom = NdotH2 * (a2 - 1.0) + 1.0;
		return a2 * math::RECIPROCAL_PI / (denom * denom);
	}

	float Lambda(in float NdotX2)
	{
		// TODO
		// ggx is a little special...
		return 0.f / 0.f;//nan
	}
};

struct GGX : IsotropicGGX
{
	float ay;
	float ay2;

	static GGX create(float _ax, float _ay, float _ax2, float _ay2)
	{
		GGX b;
		b.a = _ax;
		b.ay = _ay;
		b.a2 = _ax2;
		b.ay2 = _ay2;
		return b;
	}
	static GGX create(float _ax, float _ay)
	{
		return create(_ax, _ay, _ax*_ax, _ay*_ay);
	}

	template <class IncomingrayDirInfo>
	float3 generateH(in surface_interactions::Anisotropic<IncomingrayDirInfo> interaction, inout float3 u, out AnisotropicMicrofacetCache cache)
	{
		return generateH_impl(interaction, u, cache, a, ay);
	}

	float D(in float TdotH2, in float BdotH2, in float NdotH2)
	{
		float aa = a * ay;
		float denom = TdotH2 / a2 + BdotH2 / ay2 + NdotH2;
		return math::RECIPROCAL_PI / (aa * denom * denom);
	}

	float Lambda(in float TdotH2, in float BdotH2, in float NdotX2)
	{
		// TODO
		// ggx is a little special...
		return 0.f / 0.f; //nan
	}
};

}
}
}
}
}



#endif