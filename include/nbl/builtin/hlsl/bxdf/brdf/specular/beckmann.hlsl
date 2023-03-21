// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_BECKMANN_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/reflection.hlsl>
#include <nbl/builtin/hlsl/bxdf/fresnel.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf/beckmann.hlsl>
#include <nbl/builtin/hlsl/bxdf/geom/smith/common.hlsl>
#include <nbl/builtin/hlsl/bxdf/geom/smith/beckmann.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>

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

float3 beckmann_cos_generate_wo_clamps(in float3 localV, in float2 u, in float ax, in float ay)
{
    //stretch
    float3 V = normalize(float3(ax*localV.x, ay*localV.y, localV.z));

    float2 slope;
    if (V.z > 0.9999)//V.z=NdotV=cosTheta in tangent space
    {
        float r = sqrt(-log(1.0-u.x));
        float sinPhi = sin(2.0*math::PI*u.y);
        float cosPhi = cos(2.0*math::PI*u.y);
        slope = float2(r,r)*float2(cosPhi,sinPhi);
    }
    else
    {
        float cosTheta = V.z;
        float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
        float tanTheta = sinTheta/cosTheta;
        float cotTheta = 1.0/tanTheta;
        
        float a = -1.0;
        float c = math::erf(cosTheta);
        float sample_x = max(u.x, 1.0e-6f);
        float theta = acos(cosTheta);
        float fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594*theta));
        float b = c - (1.0 + c) * pow(1.0-sample_x, fit);
        
        float normalization = 1.0 / (1.0 + c + math::SQRT_RECIPROCAL_PI * tanTheta * exp(-cosTheta*cosTheta));

        const int ITER_THRESHOLD = 10;
		const float MAX_ACCEPTABLE_ERR = 1.0e-5;
        int it = 0;
        float value=1000.0;
        while (++it<ITER_THRESHOLD && abs(value)>MAX_ACCEPTABLE_ERR)
        {
            if (!(b>=a && b<=c))
                b = 0.5 * (a+c);

            float invErf = math::erfInv(b);
            value = normalization * (1.0 + b + math::SQRT_RECIPROCAL_PI * tanTheta * exp(-invErf*invErf)) - sample_x;
            float derivative = normalization * (1.0 - invErf*cosTheta);

            if (value > 0.0)
                c = b;
            else
                a = b;

            b -= value/derivative;
        }
        // TODO: investigate if we can replace these two erf^-1 calls with a box muller transform
        slope.x = math::erfInv(b);
        slope.y = math::erfInv(2.0f * max(u.y,1.0e-6f) - 1.0f);
    }
    
    float sinTheta = sqrt(1.0f - V.z*V.z);
    float cosPhi = sinTheta==0.0f ? 1.0f : clamp(V.x/sinTheta, -1.0f, 1.0f);
    float sinPhi = sinTheta==0.0f ? 0.0f : clamp(V.y/sinTheta, -1.0f, 1.0f);
    //rotate
    float tmp = cosPhi*slope.x - sinPhi*slope.y;
    slope.y = sinPhi*slope.x + cosPhi*slope.y;
    slope.x = tmp;

    //unstretch
    slope = float2(ax,ay)*slope;

    return normalize(float3(-slope, 1.0));
}

// TODO: unifty the two following functions into `microfacet_BRDF_cos_generate_wo_clamps(float3 H,...)` and `microfacet_BRDF_cos_generate` or at least a auto declaration macro in lieu of a template
template <class RayDirInfo>
LightSample<RayDirInfo> beckmann_cos_generate_wo_clamps(in float3 localV, in float3x3 m, in float2 u, in float ax, in float ay, out AnisotropicMicrofacetCache _cache)
{
    const float3 H = beckmann_cos_generate_wo_clamps(localV,u,ax,ay);
    
    _cache = AnisotropicMicrofacetCache::create(localV,H);
    float3 localL = math::reflect(localV, H, _cache.VdotH);
    
    return LightSample<RayDirInfo>::createTangentSpace(localV, RayDirInfo::create(localL), m);
}

template <class RayDirInfo>
LightSample<RayDirInfo> beckmann_cos_generate(in surface_interactions::Anisotropic<RayDirInfo> interaction, in float2 u, in float ax, in float ay, out AnisotropicMicrofacetCache _cache)
{
    const float3 localV = interaction.getTangentSpaceV();
    const float3x3 m = interaction.getTangentFrame();
    return beckmann_cos_generate_wo_clamps<RayDirInfo>(localV,m,u,ax,ay,_cache);
}



// isotropic PDF
float beckmann_pdf_wo_clamps(in float ndf, in float maxNdotV, in float NdotV2, in float a2, out float onePlusLambda_V)
{
    const float lambda = geom_smith::beckmann::Lambda(NdotV2, a2);
    return geom_smith::VNDF_pdf_wo_clamps(ndf,lambda,maxNdotV,onePlusLambda_V);
}

float beckmann_pdf_wo_clamps(in float NdotH2, in float maxNdotV, in float NdotV2, in float a2)
{
    float ndf = ndf::beckmann(a2, NdotH2);

    float dummy;
    return beckmann_pdf_wo_clamps(ndf, maxNdotV,NdotV2, a2, dummy);
}

// anisotropic PDF
float beckmann_pdf_wo_clamps(in float ndf, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float ax2, in float ay2, out float onePlusLambda_V)
{
    float c2 = geom_smith::beckmann::C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
    float lambda = geom_smith::beckmann::Lambda(c2);

    return geom_smith::VNDF_pdf_wo_clamps(ndf, lambda, maxNdotV, onePlusLambda_V);
}

float beckmann_pdf_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float ax, in float ax2, in float ay, in float ay2)
{
    float ndf = ndf::beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);

    float dummy;
    return beckmann_pdf_wo_clamps(ndf, maxNdotV, TdotV2, BdotV2, NdotV2, ax2, ay2, dummy);
}

quotient_and_pdf_rgb beckmann_cos_quotient_and_pdf_wo_clamps(in float ndf, in float NdotL2, in float maxNdotV, in float NdotV2, in float3 reflectance, in float a2)
{
    float onePlusLambda_V;
    float pdf = beckmann_pdf_wo_clamps(ndf,maxNdotV,NdotV2,a2,onePlusLambda_V);

    float G2_over_G1 = geom_smith::beckmann::G2_over_G1(onePlusLambda_V, NdotL2, a2);
    return quotient_and_pdf_rgb::create(reflectance*G2_over_G1, pdf);
}
template <class RayDirInfo>
quotient_and_pdf_rgb beckmann_cos_quotient_and_pdf(in LightSample<RayDirInfo> _sample, in surface_interactions::Isotropic<RayDirInfo> interaction, in IsotropicMicrofacetCache _cache, in float2x3 ior, in float a2)
{
    const float ndf = ndf::beckmann(a2, _cache.NdotH2);
    float onePlusLambda_V;
    float pdf = beckmann_pdf_wo_clamps(ndf, interaction.NdotV, interaction.NdotV2, a2, onePlusLambda_V);
    float3 rem = float3(0.0,0.0,0.0);
    if (_sample.NdotL>FLT_MIN && interaction.NdotV>FLT_MIN)
    {
        const float3 reflectance = fresnel::conductor(ior[0], ior[1], _cache.VdotH);
    
        float G2_over_G1 = beckmann_smith_G2_over_G1(onePlusLambda_V, _sample.NdotL2, a2);
        rem = reflectance * G2_over_G1;
    }
    
    return quotient_and_pdf_rgb::create(rem, pdf);
}



quotient_and_pdf_rgb beckmann_aniso_cos_quotient_and_pdf_wo_clamps(in float ndf, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float3 reflectance, in float ax2, in float ay2)
{
    float onePlusLambda_V;
    float pdf = beckmann_pdf_wo_clamps(ndf,maxNdotV,TdotV2,BdotV2,NdotV2,ax2,ay2,onePlusLambda_V);

    float G2_over_G1 = geom_smith::beckmann::G2_over_G1(onePlusLambda_V, TdotL2, BdotL2, NdotL2, ax2, ay2);
    return quotient_and_pdf_rgb::create(reflectance * G2_over_G1, pdf);
}
template <class RayDirInfo>
quotient_and_pdf_rgb beckmann_aniso_cos_quotient_and_pdf(in LightSample<RayDirInfo> _sample, in surface_interactions::Anisotropic<RayDirInfo> interaction, in AnisotropicMicrofacetCache _cache, in float2x3 ior, in float ax, in float ay)
{    
    const float ax2 = ax * ax;
    const float ay2 = ay * ay;

    const float TdotH2 = _cache.TdotH * _cache.TdotH;
    const float BdotH2 = _cache.BdotH * _cache.BdotH;
    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;

    const float NdotV2 = interaction.NdotV2;

    const float ndf = ndf::beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, _cache.NdotH2);
    float onePlusLambda_V;
    float pdf = beckmann_pdf_wo_clamps(ndf, interaction.NdotV, TdotV2, BdotV2, NdotV2, ax2, ay2, onePlusLambda_V);
    quotient_and_pdf_rgb qpdf = quotient_and_pdf_rgb::create(float3(0.0, 0.0, 0.0), pdf);
    if (_sample.NdotL>FLT_MIN && interaction.NdotV>FLT_MIN)
    {
        const float TdotL2 = _sample.TdotL*_sample.TdotL;
        const float BdotL2 = _sample.BdotL*_sample.BdotL;
    
        const float3 reflectance = fresnel::conductor(ior[0], ior[1], _cache.VdotH);

        qpdf = beckmann_aniso_cos_quotient_and_pdf_wo_clamps(ndf, _sample.NdotL2, TdotL2, BdotL2, interaction.NdotV, TdotV2, BdotV2, NdotV2, reflectance, ax2, ay2);
    }
    
    return qpdf;
}


float beckmann_height_correlated_cos_eval_DG_wo_clamps(in float NdotH2, in float NdotL2, in float NdotV2, in float a2)
{
    float NG = ndf::beckmann(a2, NdotH2);
    if  (a2>FLT_MIN)
        NG *= geom_smith::beckmann::correlated(NdotV2, NdotL2, a2);
    
    return NG;
}
float3 beckmann_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in float2x3 ior, in float a2)
{
    const float NG = beckmann_height_correlated_cos_eval_DG_wo_clamps(NdotH2, NdotL2, NdotV2, a2);

    const float3 fr = fresnel::conductor(ior[0], ior[1], VdotH);

    return fr*ndf::microfacet_to_light_measure_transform(NG,maxNdotV);
}
template <class RayDirInfo>
float3 beckmann_height_correlated_cos_eval(in LightSample<RayDirInfo> _sample, in surface_interactions::Isotropic<RayDirInfo> interaction, in IsotropicMicrofacetCache _cache, in float2x3 ior, in float a2)
{
    if (interaction.NdotV>FLT_MIN)
        return beckmann_height_correlated_cos_eval_wo_clamps(_cache.NdotH2,_sample.NdotL2,interaction.NdotV,interaction.NdotV2,_cache.VdotH,ior,a2);
    else
        return float3(0.0,0.0,0.0);
}

float beckmann_aniso_height_correlated_cos_eval_DG_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float NdotL2, in float TdotL2, in float BdotL2, in float NdotV2, in float TdotV2, in float BdotV2, in float ax, in float ax2, in float ay, in float ay2)
{
    float NG = ndf::beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);
    if (ax>FLT_MIN || ay>FLT_MIN)
        NG *= geom_smith::beckmann::correlated(TdotV2, BdotV2, NdotV2, TdotL2, BdotL2, NdotL2, ax2, ay2);
    
    return NG;
}
float3 beckmann_aniso_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float VdotH, in float2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    const float NG = beckmann_aniso_height_correlated_cos_eval_DG_wo_clamps(NdotH2,TdotH2,BdotH2, NdotL2,TdotL2,BdotL2, NdotV2,TdotV2,BdotV2, ax, ax2, ay, ay2);

    const float3 fr = fresnel::conductor(ior[0], ior[1], VdotH);
    
    return fr*ndf::microfacet_to_light_measure_transform(NG,maxNdotV);
}
template <class RayDirInfo>
float3 beckmann_aniso_height_correlated_cos_eval(in LightSample<RayDirInfo> _sample, in surface_interactions::Isotropic<RayDirInfo> interaction, in AnisotropicMicrofacetCache _cache, in float2x3 ior, in float ax, in float ay)
{    
    if (interaction.NdotV>FLT_MIN)
    {
        const float TdotH2 = _cache.TdotH*_cache.TdotH;
        const float BdotH2 = _cache.BdotH*_cache.BdotH;

        const float TdotL2 = _sample.TdotL*_sample.TdotL;
        const float BdotL2 = _sample.BdotL*_sample.BdotL;

        const float TdotV2 = interaction.TdotV*interaction.TdotV;
        const float BdotV2 = interaction.BdotV*interaction.BdotV;
        return beckmann_aniso_height_correlated_cos_eval_wo_clamps(_cache.NdotH2,TdotH2,BdotH2, _sample.NdotL2,TdotL2,BdotL2, interaction.NdotV,interaction.NdotV2,TdotV2,BdotV2, _cache.VdotH, ior,ax,ax*ax,ay,ay*ay);
    }
    else
    {
        return float3(0.0, 0.0, 0.0);
    }
}

}
}
}
}
}

#endif
