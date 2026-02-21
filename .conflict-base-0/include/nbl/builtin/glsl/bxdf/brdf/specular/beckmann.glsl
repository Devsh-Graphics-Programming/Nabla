// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_GLSL_BXDF_BRDF_SPECULAR_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_GLSL_BXDF_BRDF_SPECULAR_BECKMANN_INCLUDED_

#include <nbl/builtin/glsl/bxdf/common_samples.glsl>
#include <nbl/builtin/glsl/bxdf/fresnel.glsl>
#include <nbl/builtin/glsl/bxdf/ndf/beckmann.glsl>
#include <nbl/builtin/glsl/bxdf/geom/smith/common.glsl>
#include <nbl/builtin/glsl/bxdf/geom/smith/beckmann.glsl>
#include <nbl/builtin/glsl/math/functions.glsl>

#include <nbl/builtin/glsl/math/functions.glsl>

vec3 nbl_glsl_beckmann_cos_generate_wo_clamps(in vec3 localV, in vec2 u, in float ax, in float ay)
{
    //stretch
    vec3 V = normalize(vec3(ax*localV.x, ay*localV.y, localV.z));

    vec2 slope;
    if (V.z > 0.9999)//V.z=NdotV=cosTheta in tangent space
    {
        float r = sqrt(-log(1.0-u.x));
        float sinPhi = sin(2.0*nbl_glsl_PI*u.y);
        float cosPhi = cos(2.0*nbl_glsl_PI*u.y);
        slope = vec2(r)*vec2(cosPhi,sinPhi);
    }
    else
    {
        float cosTheta = V.z;
        float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
        float tanTheta = sinTheta/cosTheta;
        float cotTheta = 1.0/tanTheta;
        
        float a = -1.0;
        float c = nbl_glsl_erf(cosTheta);
        float sample_x = max(u.x, 1.0e-6);
        float theta = acos(cosTheta);
        float fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594*theta));
        float b = c - (1.0 + c) * pow(1.0-sample_x, fit);
        
        float normalization = 1.0 / (1.0 + c + nbl_glsl_SQRT_RECIPROCAL_PI * tanTheta * exp(-cosTheta*cosTheta));

        const int ITER_THRESHOLD = 10;
		const float MAX_ACCEPTABLE_ERR = 1.0e-5;
        int it = 0;
        float value=1000.0;
        while (++it<ITER_THRESHOLD && abs(value)>MAX_ACCEPTABLE_ERR)
        {
            if (!(b>=a && b<=c))
                b = 0.5 * (a+c);

            float invErf = nbl_glsl_erfInv(b);
            value = normalization * (1.0 + b + nbl_glsl_SQRT_RECIPROCAL_PI * tanTheta * exp(-invErf*invErf)) - sample_x;
            float derivative = normalization * (1.0 - invErf*cosTheta);

            if (value > 0.0)
                c = b;
            else
                a = b;

            b -= value/derivative;
        }
        // TODO: investigate if we can replace these two erf^-1 calls with a box muller transform
        slope.x = nbl_glsl_erfInv(b);
        slope.y = nbl_glsl_erfInv(2.0 * max(u.y,1.0e-6) - 1.0);
    }
    
    float sinTheta = sqrt(1.0 - V.z*V.z);
    float cosPhi = sinTheta==0.0 ? 1.0 : clamp(V.x/sinTheta, -1.0, 1.0);
    float sinPhi = sinTheta==0.0 ? 0.0 : clamp(V.y/sinTheta, -1.0, 1.0);
    //rotate
    float tmp = cosPhi*slope.x - sinPhi*slope.y;
    slope.y = sinPhi*slope.x + cosPhi*slope.y;
    slope.x = tmp;

    //unstretch
    slope = vec2(ax,ay)*slope;

    return normalize(vec3(-slope, 1.0));
}

// TODO: unifty the two following functions into `nbl_glsl_microfacet_BRDF_cos_generate_wo_clamps(vec3 H,...)` and `nbl_glsl_microfacet_BRDF_cos_generate` or at least a auto declaration macro in lieu of a template
nbl_glsl_LightSample nbl_glsl_beckmann_cos_generate_wo_clamps(in vec3 localV, in mat3 m, in vec2 u, in float ax, in float ay, out nbl_glsl_AnisotropicMicrofacetCache _cache)
{
    const vec3 H = nbl_glsl_beckmann_cos_generate_wo_clamps(localV,u,ax,ay);
    
    vec3 localL;
    _cache = nbl_glsl_calcAnisotropicMicrofacetCache(localV,H,localL);
    
    return nbl_glsl_createLightSampleTangentSpace(localV,localL,m);
}

nbl_glsl_LightSample nbl_glsl_beckmann_cos_generate(in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u, in float ax, in float ay, out nbl_glsl_AnisotropicMicrofacetCache _cache)
{
    const vec3 localV = nbl_glsl_getTangentSpaceV(interaction);
    const mat3 m = nbl_glsl_getTangentFrame(interaction);
    return nbl_glsl_beckmann_cos_generate_wo_clamps(localV,m,u,ax,ay,_cache);
}



// isotropic PDF
float nbl_glsl_beckmann_pdf_wo_clamps(in float ndf, in float maxNdotV, in float NdotV2, in float a2, out float onePlusLambda_V)
{
    const float lambda = nbl_glsl_smith_beckmann_Lambda(NdotV2, a2);
    return nbl_glsl_smith_VNDF_pdf_wo_clamps(ndf,lambda,maxNdotV,onePlusLambda_V);
}

float nbl_glsl_beckmann_pdf_wo_clamps(in float NdotH2, in float maxNdotV, in float NdotV2, in float a2)
{
    float ndf = nbl_glsl_beckmann(a2, NdotH2);

    float dummy;
    return nbl_glsl_beckmann_pdf_wo_clamps(ndf, maxNdotV,NdotV2, a2, dummy);
}

// anisotropic PDF
float nbl_glsl_beckmann_pdf_wo_clamps(in float ndf, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float ax2, in float ay2, out float onePlusLambda_V)
{
    float c2 = nbl_glsl_smith_beckmann_C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
    float lambda = nbl_glsl_smith_beckmann_Lambda(c2);

    return nbl_glsl_smith_VNDF_pdf_wo_clamps(ndf, lambda, maxNdotV, onePlusLambda_V);
}

float nbl_glsl_beckmann_pdf_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float ax, in float ax2, in float ay, in float ay2)
{
    float ndf = nbl_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);

    float dummy;
    return nbl_glsl_beckmann_pdf_wo_clamps(ndf, maxNdotV, TdotV2, BdotV2, NdotV2, ax2, ay2, dummy);
}

vec3 nbl_glsl_beckmann_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float NdotL2, in float maxNdotV, in float NdotV2, in vec3 reflectance, in float a2)
{
    float onePlusLambda_V;
    pdf = nbl_glsl_beckmann_pdf_wo_clamps(ndf,maxNdotV,NdotV2,a2,onePlusLambda_V);

    float G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(onePlusLambda_V, NdotL2, a2);
    return reflectance*G2_over_G1;
}
vec3 nbl_glsl_beckmann_cos_remainder_and_pdf(out float pdf, in nbl_glsl_LightSample _sample, in nbl_glsl_IsotropicViewSurfaceInteraction interaction, in nbl_glsl_IsotropicMicrofacetCache _cache, in mat2x3 ior, in float a2)
{
    const float ndf = nbl_glsl_beckmann(a2, _cache.NdotH2);
    float onePlusLambda_V;
    pdf = nbl_glsl_beckmann_pdf_wo_clamps(ndf, interaction.NdotV, interaction.NdotV_squared, a2, onePlusLambda_V);
    vec3 rem = vec3(0.0);
    if (_sample.NdotL>nbl_glsl_FLT_MIN && interaction.NdotV>nbl_glsl_FLT_MIN)
    {
        const vec3 reflectance = nbl_glsl_fresnel_conductor(ior[0], ior[1], _cache.VdotH);
    
        float G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(onePlusLambda_V, _sample.NdotL2, a2);
        rem = reflectance * G2_over_G1;
    }
    
    return rem;
}



vec3 nbl_glsl_beckmann_aniso_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in vec3 reflectance, in float ax2, in float ay2)
{
    float onePlusLambda_V;
    pdf = nbl_glsl_beckmann_pdf_wo_clamps(ndf,maxNdotV,TdotV2,BdotV2,NdotV2,ax2,ay2,onePlusLambda_V);

    float G2_over_G1 = nbl_glsl_beckmann_smith_G2_over_G1(onePlusLambda_V, TdotL2, BdotL2, NdotL2, ax2, ay2);
    return reflectance * G2_over_G1;
}
vec3 nbl_glsl_beckmann_aniso_cos_remainder_and_pdf(out float pdf, in nbl_glsl_LightSample _sample, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in nbl_glsl_AnisotropicMicrofacetCache _cache, in mat2x3 ior, in float ax, in float ay)
{    
    const float ax2 = ax * ax;
    const float ay2 = ay * ay;

    const float TdotH2 = _cache.TdotH * _cache.TdotH;
    const float BdotH2 = _cache.BdotH * _cache.BdotH;
    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;

    const float NdotV2 = interaction.isotropic.NdotV_squared;

    const float ndf = nbl_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, _cache.isotropic.NdotH2);
    float onePlusLambda_V;
    pdf = nbl_glsl_beckmann_pdf_wo_clamps(ndf, interaction.isotropic.NdotV, TdotV2, BdotV2, NdotV2, ax2, ay2, onePlusLambda_V);
    vec3 rem = vec3(0.0);
    if (_sample.NdotL>nbl_glsl_FLT_MIN && interaction.isotropic.NdotV>nbl_glsl_FLT_MIN)
    {
        const float TdotL2 = _sample.TdotL*_sample.TdotL;
        const float BdotL2 = _sample.BdotL*_sample.BdotL;
    
        const vec3 reflectance = nbl_glsl_fresnel_conductor(ior[0], ior[1], _cache.isotropic.VdotH);

	    rem = nbl_glsl_beckmann_aniso_cos_remainder_and_pdf_wo_clamps(pdf, ndf, _sample.NdotL2, TdotL2, BdotL2, interaction.isotropic.NdotV, TdotV2, BdotV2, NdotV2, reflectance, ax2, ay2);
    }
    
    return rem;
}


float nbl_glsl_beckmann_height_correlated_cos_eval_DG_wo_clamps(in float NdotH2, in float NdotL2, in float NdotV2, in float a2)
{
    float NG = nbl_glsl_beckmann(a2, NdotH2);
    if  (a2>nbl_glsl_FLT_MIN)
        NG *= nbl_glsl_beckmann_smith_correlated(NdotV2, NdotL2, a2);
    
    return NG;
}
vec3 nbl_glsl_beckmann_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    const float NG = nbl_glsl_beckmann_height_correlated_cos_eval_DG_wo_clamps(NdotH2, NdotL2, NdotV2, a2);

    const vec3 fr = nbl_glsl_fresnel_conductor(ior[0], ior[1], VdotH);

    return fr*nbl_glsl_microfacet_to_light_measure_transform(NG,maxNdotV);
}
vec3 nbl_glsl_beckmann_height_correlated_cos_eval(in nbl_glsl_LightSample _sample, in nbl_glsl_IsotropicViewSurfaceInteraction interaction, in nbl_glsl_IsotropicMicrofacetCache _cache, in mat2x3 ior, in float a2)
{
    if (interaction.NdotV>nbl_glsl_FLT_MIN)
        return nbl_glsl_beckmann_height_correlated_cos_eval_wo_clamps(_cache.NdotH2,_sample.NdotL2,interaction.NdotV,interaction.NdotV_squared,_cache.VdotH,ior,a2);
    else
        return vec3(0.0);
}

float nbl_glsl_beckmann_aniso_height_correlated_cos_eval_DG_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float NdotL2, in float TdotL2, in float BdotL2, in float NdotV2, in float TdotV2, in float BdotV2, in float ax, in float ax2, in float ay, in float ay2)
{
    float NG = nbl_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);
    if (ax>nbl_glsl_FLT_MIN || ay>nbl_glsl_FLT_MIN)
        NG *= nbl_glsl_beckmann_smith_correlated(TdotV2, BdotV2, NdotV2, TdotL2, BdotL2, NdotL2, ax2, ay2);
    
    return NG;
}
vec3 nbl_glsl_beckmann_aniso_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float VdotH, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    const float NG = nbl_glsl_beckmann_aniso_height_correlated_cos_eval_DG_wo_clamps(NdotH2,TdotH2,BdotH2, NdotL2,TdotL2,BdotL2, NdotV2,TdotV2,BdotV2, ax, ax2, ay, ay2);

    const vec3 fr = nbl_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    
    return fr*nbl_glsl_microfacet_to_light_measure_transform(NG,maxNdotV);
}
vec3 nbl_glsl_beckmann_aniso_height_correlated_cos_eval(in nbl_glsl_LightSample _sample, in nbl_glsl_AnisotropicViewSurfaceInteraction interaction, in nbl_glsl_AnisotropicMicrofacetCache _cache, in mat2x3 ior, in float ax, in float ay)
{    
    if (interaction.isotropic.NdotV>nbl_glsl_FLT_MIN)
    {
        const float TdotH2 = _cache.TdotH*_cache.TdotH;
        const float BdotH2 = _cache.BdotH*_cache.BdotH;

        const float TdotL2 = _sample.TdotL*_sample.TdotL;
        const float BdotL2 = _sample.BdotL*_sample.BdotL;

        const float TdotV2 = interaction.TdotV*interaction.TdotV;
        const float BdotV2 = interaction.BdotV*interaction.BdotV;
      return nbl_glsl_beckmann_aniso_height_correlated_cos_eval_wo_clamps(_cache.isotropic.NdotH2,TdotH2,BdotH2, _sample.NdotL2,TdotL2,BdotL2, interaction.isotropic.NdotV,interaction.isotropic.NdotV_squared,TdotV2,BdotV2, _cache.isotropic.VdotH, ior,ax,ax*ax,ay,ay*ay);
    }
    else
        return vec3(0.0);
}

#endif
