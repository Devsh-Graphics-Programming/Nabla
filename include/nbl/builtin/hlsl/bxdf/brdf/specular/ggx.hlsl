// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_BRDF_SPECULAR_GGX_INCLUDED_

#include <nbl/builtin/hlsl/bxdf/reflection.hlsl>
#include <nbl/builtin/hlsl/bxdf/fresnel.hlsl>
#include <nbl/builtin/hlsl/bxdf/ndf/ggx.hlsl>
#include <nbl/builtin/hlsl/bxdf/geom/smith/common.hlsl>
#include <nbl/builtin/hlsl/bxdf/geom/smith/ggx.hlsl>

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

//depr
/*
 float3 ggx_height_correlated_aniso_cos_eval(in BSDFAnisotropicParams params, in surface_interactions::Anisotropic<RayDirInfo> inter, in float2x3 ior, in float a2, in float2 atb, in float aniso)
{
    float g = geom_smith::ggx::height_correlated_aniso_wo_numerator(atb.x, atb.y, params.TdotL, interaction.TdotV, params.BdotL, interaction.BdotV, params.NdotL, interaction.NdotV);
    float ndf = ggx_burley_aniso(aniso, a2, params.TdotH, params.BdotH, params.NdotH);
    float3 fr = fresnel_conductor(ior[0], ior[1], params.VdotH);

    return params.NdotL * g*ndf*fr;
}
*/
//defined using NDF function with better API (compared to burley used above) and new impl of correlated smith
 float ggx_height_correlated_aniso_cos_eval_DG_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float ax, in float ax2, in float ay, in float ay2)
{
    float NG = ndf::ggx::aniso(TdotH2,BdotH2,NdotH2, ax, ay, ax2, ay2);
    if (ax>FLT_MIN || ay>FLT_MIN)
    {
        NG *= geom_smith::ggx::correlated_wo_numerator(
            maxNdotV, TdotV2, BdotV2, NdotV2,
            maxNdotL, TdotL2, BdotL2, NdotL2,
            ax2, ay2
        );
    }

    return NG;
}

 float3 ggx_height_correlated_aniso_cos_eval_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float VdotH, in float2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    float NG_already_in_reflective_dL_measure = ggx_height_correlated_aniso_cos_eval_DG_wo_clamps(NdotH2,TdotH2,BdotH2,maxNdotL,NdotL2,TdotL2,BdotL2,maxNdotV,NdotV2,TdotV2,BdotV2,ax,ax2,ay,ay2);

    float3 fr = fresnel::conductor(ior[0], ior[1], VdotH);
    return fr*ndf::ggx::microfacet_to_light_measure_transform(NG_already_in_reflective_dL_measure,maxNdotL);
}

 template <class RayDirInfo>
 float3 ggx_height_correlated_aniso_cos_eval(in LightSample<RayDirInfo> _sample, in surface_interactions::Anisotropic<RayDirInfo> interaction, in AnisotropicMicrofacetCache _cache, in float2x3 ior, in float ax, in float ay)
{
    if (_sample.NdotL>FLT_MIN && interaction.NdotV>FLT_MIN)
    {
        const float TdotH2 = _cache.TdotH*_cache.TdotH;
        const float BdotH2 = _cache.BdotH*_cache.BdotH;

        const float TdotL2 = _sample.TdotL*_sample.TdotL;
        const float BdotL2 = _sample.BdotL*_sample.BdotL;

        const float TdotV2 = interaction.TdotV*interaction.TdotV;
        const float BdotV2 = interaction.BdotV*interaction.BdotV;
        return ggx_height_correlated_aniso_cos_eval_wo_clamps(_cache.NdotH2, TdotH2, BdotH2, _sample.NdotL,_sample.NdotL2,TdotL2,BdotL2, interaction.NdotV,interaction.NdotV_squared,TdotV2,BdotV2, _cache.VdotH, ior, ax,ax*ax,ay,ay*ay);
    }
    else
        return float3(0.0,0.0,0.0);
}


 float ggx_height_correlated_cos_eval_DG_wo_clamps(in float NdotH2, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in float a2)
{
    float NG = ndf::ggx::trowbridge_reitz(a2, NdotH2);
    if (a2>FLT_MIN)
        NG *= geom_smith::ggx::correlated_wo_numerator(maxNdotV, NdotV2, maxNdotL, NdotL2, a2);

    return NG;
}

 float3 ggx_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in float2x3 ior, in float a2)
{
    float NG_already_in_reflective_dL_measure = ggx_height_correlated_cos_eval_DG_wo_clamps(NdotH2, maxNdotL, NdotL2, maxNdotV, NdotV2, a2);

    float3 fr = fresnel::conductor(ior[0], ior[1], VdotH);

    return fr*ndf::ggx::microfacet_to_light_measure_transform(NG_already_in_reflective_dL_measure, maxNdotL);
}

 template <class RayDirInfo>
 float3 ggx_height_correlated_cos_eval(in LightSample<RayDirInfo> _sample, in surface_interactions::Isotropic<RayDirInfo> interaction, in IsotropicMicrofacetCache _cache, in float2x3 ior, in float a2)
{
    if (_sample.NdotL>FLT_MIN && interaction.NdotV>FLT_MIN)
        return ggx_height_correlated_cos_eval_wo_clamps(_cache.NdotH2,max(_sample.NdotL,0.0),_sample.NdotL2, max(interaction.NdotV,0.0), interaction.NdotV_squared, _cache.VdotH,ior,a2);
    else
        return float3(0.0,0.0,0.0);
}



//Heitz's 2018 paper "Sampling the GGX Distribution of Visible Normals"
 float3 ggx_cos_generate(in float3 localV, in float2 u, in float _ax, in float _ay)
{
    float3 V = normalize(float3(_ax*localV.x, _ay*localV.y, localV.z));//stretch view vector so that we're sampling as if roughness=1.0

    float lensq = V.x*V.x + V.y*V.y;
    float3 T1 = lensq > 0.0 ? float3(-V.y, V.x, 0.0)*rsqrt(lensq) : float3(1.0,0.0,0.0);
    float3 T2 = cross(V,T1);

    float r = sqrt(u.x);
    float phi = 2.0 * math::PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + V.z);
    t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
    
    //reprojection onto hemisphere
	//TODO try it wothout the& max(), not sure if -t1*t1-t2*t2>-1.0
    float3 H = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0-t1*t1-t2*t2))*V;
    //unstretch
    return normalize(float3(_ax*H.x, _ay*H.y, H.z));
}

// TODO: unifty the two following functions into `microfacet_BRDF_cos_generate_wo_clamps(float3 H,...)` and `microfacet_BRDF_cos_generate` or at least a auto declaration macro in lieu of a template
 template <class RayDirInfo>
 LightSample<RayDirInfo> ggx_cos_generate_wo_clamps(in float3 localV, in float3x3 m, in float2 u, in float _ax, in float _ay, out AnisotropicMicrofacetCache _cache)
{
    const float3 H = ggx_cos_generate(localV,u,_ax,_ay);
    
    _cache = AnisotropicMicrofacetCache::create(localV, H);
    float3 localL;
    localL = math::reflect(localV, H, _cache.VdotH);
    
    return LightSample<RayDirInfo>::createTangentSpace(localV, RayDirInfo::create(localL), m);
}

 template <class RayDirInfo>
 LightSample<RayDirInfo> ggx_cos_generate(in surface_interactions::Anisotropic<RayDirInfo> interaction, in float2 u, in float _ax, in float _ay, out AnisotropicMicrofacetCache _cache)
{
    const float3 localV = interaction.getTangentSpaceV();
    const float3x3 m = interaction.getTangentFrame();
    return ggx_cos_generate_wo_clamps<RayDirInfo>(localV,m,u,_ax,_ay,_cache);
}



 float ggx_pdf_wo_clamps(in float ndf, in float devsh_v, in float maxNdotV)
{
    return geom_smith::VNDF_pdf_wo_clamps(ndf, geom_smith::ggx::G1_wo_numerator(maxNdotV,devsh_v));
}
 float ggx_pdf_wo_clamps(in float NdotH2, in float maxNdotV, in float NdotV2, in float a2)
{
    const float ndf = ndf::ggx::trowbridge_reitz(a2, NdotH2);
    const float devsh_v = geom_smith::ggx::devsh_part(NdotV2, a2, 1.0-a2);

    return ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);
}

 float ggx_pdf_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float ax, in float ay, in float ax2, in float ay2)
{
    const float ndf = ndf::ggx::aniso(TdotH2,BdotH2,NdotH2, ax, ay, ax2, ay2);
    const float devsh_v = geom_smith::ggx::devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);

    return ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);
}


 quotient_and_pdf_rgb ggx_cos_quotient_and_pdf_wo_clamps(in float ndf, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in float3 reflectance, in float a2)
{
    const float one_minus_a2 = 1.0 - a2;
    const float devsh_v = geom_smith::ggx::devsh_part(NdotV2, a2, one_minus_a2);
    const float pdf = ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);

    const float G2_over_G1 = geom_smith::ggx::G2_over_G1_devsh(maxNdotL, NdotL2, maxNdotV, devsh_v, a2, one_minus_a2);

    return quotient_and_pdf_rgb::create(reflectance * G2_over_G1, pdf);
}

 template <class RayDirInfo>
 quotient_and_pdf_rgb ggx_cos_quotient_and_pdf(in LightSample<RayDirInfo> _sample, in surface_interactions::Isotropic<RayDirInfo> interaction, in IsotropicMicrofacetCache _cache, in float2x3 ior, in float a2)
{    
    const float one_minus_a2 = 1.0 - a2;
    const float ndf = ndf::ggx::trowbridge_reitz(a2, _cache.NdotH2);
    const float devsh_v = geom_smith::ggx::devsh_part(interaction.NdotV_squared, a2, one_minus_a2);
    const float pdf = ggx_pdf_wo_clamps(ndf, devsh_v, interaction.NdotV);

    quotient_and_pdf_rgb qpdf = quotient_and_pdf_rgb::create(float3(0.0, 0.0, 0.0), pdf);
    if (_sample.NdotL>FLT_MIN && interaction.NdotV>FLT_MIN)
    {
        const float3 reflectance = fresnel::conductor(ior[0], ior[1], _cache.VdotH);
        const float G2_over_G1 = geom_smith::ggx::G2_over_G1_devsh(_sample.NdotL, _sample.NdotL2, interaction.NdotV, devsh_v, a2, one_minus_a2);

        qpdf.quotient = reflectance * G2_over_G1;
    }

    return qpdf;
}


 quotient_and_pdf_rgb ggx_aniso_cos_quotient_and_pdf_wo_clamps(in float ndf, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float3 reflectance, in float ax2,in float ay2)
{
    const float devsh_v = geom_smith::ggx::devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);
    const float pdf = ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);

    const float G2_over_G1 = geom_smith::ggx::G2_over_G1_devsh(
        maxNdotL, TdotL2,BdotL2,NdotL2,
        maxNdotV, devsh_v,
        ax2, ay2
    );

    return quotient_and_pdf_rgb::create(reflectance * G2_over_G1, pdf);
}

 template <class RayDirInfo>
 quotient_and_pdf_rgb ggx_aniso_cos_quotient_and_pdf(in LightSample<RayDirInfo> _sample, in surface_interactions::Anisotropic<RayDirInfo> interaction, in AnisotropicMicrofacetCache _cache, in float2x3 ior, in float ax, in float ay)
{
    const float ax2 = ax * ax;
    const float ay2 = ay * ay;

    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;
    const float NdotV2 = interaction.NdotV_squared;

    const float TdotH2 = _cache.TdotH * _cache.TdotH;
    const float BdotH2 = _cache.BdotH * _cache.BdotH;

    const float devsh_v = geom_smith::ggx::devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);
    const float ndf = ndf::ggx::aniso(TdotH2, BdotH2, _cache.NdotH2, ax, ay, ax2, ay2);
    const float pdf = ggx_pdf_wo_clamps(ndf, devsh_v, interaction.NdotV);
    quotient_and_pdf_rgb qpdf = quotient_and_pdf_rgb::create(float3(0.0, 0.0, 0.0), pdf);
    if (_sample.NdotL>FLT_MIN && interaction.NdotV>FLT_MIN)
    {
        const float TdotL2 = _sample.TdotL*_sample.TdotL;
        const float BdotL2 = _sample.BdotL*_sample.BdotL;

        const float3 reflectance = fresnel::conductor(ior[0], ior[1], _cache.VdotH);
        const float G2_over_G1 = geom_smith::ggx::G2_over_G1_devsh(
            _sample.NdotL, TdotL2, BdotL2, _sample.NdotL2,
            interaction.NdotV, devsh_v,
            ax2, ay2
        );

        qpdf.quotient = reflectance * G2_over_G1;
    }

    return qpdf;
}

}
}
}
}
}

#endif
