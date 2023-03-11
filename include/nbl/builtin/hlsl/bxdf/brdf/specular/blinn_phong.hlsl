// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/common.hlsl>
#include <nbl/builtin/hlsl/bxdf/reflection.hlsl>
#include <nbl/builtin/hlsl/bxdf/fresnel.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf/blinn_phong.hlsl>
#include <nbl/builtin/hlsl/bxdf/geom/smith/beckmann.hlsl>

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace brdf
{
namespace specular
{

//conversion between alpha and Phong exponent, Walter et.al.
float phong_exp_to_alpha2(in float n)
{
    return 2.0/(n+2.0);
}
//+INF for a2==0.0
float alpha2_to_phong_exp(in float a2)
{
    return 2.0/a2 - 2.0;
}

//https://zhuanlan.zhihu.com/p/58205525
//only NDF sampling
//however we dont really care about phong sampling
float3 blinn_phong_cos_generate(in float2 u, in float n)
{
    float phi = 2.0*math::PI*u.y;
    float cosTheta = pow(u.x, 1.0/(n+1.0));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    return float3(cosPhi*sinTheta, sinPhi*sinTheta, cosTheta);
}
template <class RayDirInfo>
LightSample<RayDirInfo> blinn_phong_cos_generate(in surface_interactions::Anisotropic<RayDirInfo> interaction, in float2 u, in float n, out AnisotropicMicrofacetCache _cache)
{
    const float3 H = blinn_phong_cos_generate(u,n);
    const float3 localV = interaction.getTangentSpaceV();

    _cache = AnisotropicMicrofacetCache::create(localV, H);
    float3 localL;
    localL = math::reflect(localV, H, _cache.VdotH);
    
    const float3x3 m = interaction.getTangentFrame();

    return LightSample<RayDirInfo>::createTangentSpace(localV, RayDirInfo::create(localL), m);
}

/*
float3 blinn_phong_dielectric_cos_remainder_and_pdf(out float& pdf, in BxDFSample s, in surface_interactions::Isotropic<RayDirInfo> interaction, in float n, in float3 ior)
{
	pdf = (n+1.0)*0.5*RECIPROCAL_PI * 0.25*pow(s.NdotH,n)/s.VdotH;

    float3 fr = fresnel_dielectric(ior, s.VdotH);
    return fr * s.NdotL * (n*(n + 6.0) + 8.0) * s.VdotH / ((pow(0.5,0.5*n) + n) * (n + 1.0));
}

float3 blinn_phong_conductor_cos_remainder_and_pdf(out float& pdf, in BxDFSample s, in surface_interactions::Isotropic<RayDirInfo> interaction, in float n, in float2x3 ior)
{
	pdf = (n+1.0)*0.5*RECIPROCAL_PI * 0.25*pow(s.NdotH,n)/s.VdotH;

    float3 fr = fresnel_conductor(ior[0], ior[1], s.VdotH);
    return fr * s.NdotL * (n*(n + 6.0) + 8.0) * s.VdotH / ((pow(0.5,0.5*n) + n) * (n + 1.0));
}
*/

float blinn_phong_cos_eval_DG_wo_clamps(in float NdotH, in float NdotV_squared, in float NdotL2, in float n, in float a2)
{
    float NG = blinn_phong(NdotH, n);
    if (a2>FLT_MIN)
        NG *= geom_smith::beckmann::correlated(NdotV_squared, NdotL2, a2);
    return NG;
}
float blinn_phong_cos_eval_DG_wo_clamps(in float NdotH, in float NdotV_squared, in float NdotL2, in float n)
{
    float a2 = phong_exp_to_alpha2(n);
    return blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotV_squared, NdotL2, n, a2);
}

float3 blinn_phong_cos_eval_wo_clamps(in float NdotH, in float maxNdotV, in float NdotV_squared, in float NdotL2, in float VdotH, in float n, in float2x3 ior, in float a2)
{
    float scalar_part = blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotV_squared, NdotL2, n, a2);
    return fresnel::conductor(ior[0], ior[1], VdotH)*ndf::microfacet_to_light_measure_transform(scalar_part,maxNdotV);
}
float3 blinn_phong_cos_eval_wo_clamps(in float NdotH, in float maxNdotV, in float NdotV_squared, in float NdotL2, in float VdotH, in float n, in float2x3 ior)
{
    float a2 = phong_exp_to_alpha2(n);
    return blinn_phong_cos_eval_wo_clamps(NdotH, maxNdotV, NdotV_squared, NdotL2, VdotH, n, ior, a2);
}
template <class RayDirInfo>
float3 blinn_phong_cos_eval(in LightSample<RayDirInfo> _sample, in surface_interactions::Isotropic<RayDirInfo> interaction, in IsotropicMicrofacetCache _cache, in float n, in float2x3 ior)
{
    if (interaction.NdotV>FLT_MIN)
        return blinn_phong_cos_eval_wo_clamps(_cache.NdotH, interaction.NdotV, interaction.NdotV_squared, _sample.NdotL2, _cache.VdotH, n, ior);
    else
        return float3(0.0,0.0,0.0);
}


float blinn_phong_cos_eval_DG_wo_clamps(in float NdotH, in float NdotH2, in float TdotH2, in float BdotH2, float TdotL2, float BdotL2, in float TdotV2, in float BdotV2, in float NdotV_squared, in float NdotL2, in float nx, in float ny, in float ax2, in float ay2)
{
    float DG = blinn_phong(NdotH, 1.0/(1.0-NdotH2), TdotH2, BdotH2, nx, ny);
    if (ax2>FLT_MIN || ay2>FLT_MIN)
        DG *= geom_smith::beckmann::correlated(TdotV2, BdotV2, NdotV_squared, TdotL2, BdotL2, NdotL2, ax2, ay2);
    return DG;
}
float blinn_phong_cos_eval_DG_wo_clamps(in float NdotH, in float NdotH2, in float TdotH2, in float BdotH2, in float TdotL2, in float BdotL2, in float TdotV2, in float BdotV2, in float NdotV_squared, in float NdotL2, in float nx, in float ny)
{
    float ax2 = phong_exp_to_alpha2(nx);
    float ay2 = phong_exp_to_alpha2(ny);

    return blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotH2, TdotH2, BdotH2, TdotL2, BdotL2, TdotV2, BdotV2, NdotV_squared, NdotL2, nx, ny, ax2, ay2);
}

float3 blinn_phong_cos_eval_wo_clamps(in float NdotH, in float NdotH2, in float TdotH2, in float BdotH2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV_squared, in float NdotL2, in float VdotH, in float nx, in float ny, in float2x3 ior, in float ax2, in float ay2)
{
    float scalar_part = blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotH2, TdotH2, BdotH2, TdotL2, BdotL2, TdotV2, BdotV2, NdotV_squared, NdotL2, nx, ny, ax2, ay2);

    return fresnel::conductor(ior[0], ior[1], VdotH)*ndf::microfacet_to_light_measure_transform(scalar_part,maxNdotV);
}
float3 blinn_phong_cos_eval_wo_clamps(in float NdotH, in float NdotH2, in float TdotH2, in float BdotH2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV_squared, in float NdotL2, in float VdotH, in float nx, in float ny, in float2x3 ior)
{
    float ax2 = phong_exp_to_alpha2(nx);
    float ay2 = phong_exp_to_alpha2(ny);

    return blinn_phong_cos_eval_wo_clamps(NdotH, NdotH2, TdotH2, BdotH2, TdotL2, BdotL2, maxNdotV, TdotV2, BdotV2, NdotV_squared, NdotL2, VdotH, nx, ny, ior, ax2, ay2);
}
template <class RayDirInfo>
float3 blinn_phong_cos_eval(in LightSample<RayDirInfo> _sample, in surface_interactions::Anisotropic<RayDirInfo> interaction, in AnisotropicMicrofacetCache _cache, in float nx, in float ny, in float2x3 ior)
{    
    if (interaction.NdotV>FLT_MIN)
    {
        const float TdotH2 = _cache.TdotH*_cache.TdotH;
        const float BdotH2 = _cache.BdotH*_cache.BdotH;

        const float TdotL2 = _sample.TdotL*_sample.TdotL;
        const float BdotL2 = _sample.BdotL*_sample.BdotL;

        const float TdotV2 = interaction.TdotV*interaction.TdotV;
        const float BdotV2 = interaction.BdotV*interaction.BdotV;
        return blinn_phong_cos_eval_wo_clamps(_cache.NdotH, _cache.NdotH2, TdotH2, BdotH2, TdotL2, BdotL2, interaction.NdotV, TdotV2, BdotV2, interaction.NdotV_squared, _sample.NdotL2, _cache.VdotH, nx, ny, ior);
    }
    else
        return float3(0.0,0.0,0.0);
}

}
}
}
}
}

#endif
