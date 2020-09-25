#ifndef _IRR_BUILTIN_GLSL_BXDF_BRDF_SPECULAR_GGX_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_BRDF_SPECULAR_GGX_INCLUDED_

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/fresnel.glsl>
#include <irr/builtin/glsl/bxdf/ndf/ggx.glsl>
#include <irr/builtin/glsl/bxdf/geom/smith/ggx.glsl>

//depr
/*
vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction inter, in mat2x3 ior, in float a2, in vec2 atb, in float aniso)
{
    float g = irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(atb.x, atb.y, params.TdotL, interaction.TdotV, params.BdotL, interaction.BdotV, params.isotropic.NdotL, interaction.isotropic.NdotV);
    float ndf = irr_glsl_ggx_burley_aniso(aniso, a2, params.TdotH, params.BdotH, params.isotropic.NdotH);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], params.isotropic.VdotH);

    return params.isotropic.NdotL * g*ndf*fr;
}
*/
//defined using NDF function with better API (compared to burley used above) and new impl of correlated smith
float irr_glsl_ggx_height_correlated_aniso_cos_eval_DG_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float ax, in float ax2, in float ay, in float ay2)
{
    float NG = irr_glsl_ggx_aniso(TdotH2,BdotH2,NdotH2, ax, ay, ax2, ay2);
    if (ax>FLT_MIN || ay>FLT_MIN)
    {
        NG *= irr_glsl_ggx_smith_correlated_wo_numerator(
            maxNdotV, TdotV2, BdotV2, NdotV2,
            maxNdotL, TdotL2, BdotL2, NdotL2,
            ax2, ay2
        );
    }

    return NG;
}

vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float VdotH, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    float NG_already_in_reflective_dL_measure = irr_glsl_ggx_height_correlated_aniso_cos_eval_DG_wo_clamps(NdotH2,TdotH2,BdotH2,maxNdotL,NdotL2,TdotL2,BdotL2,maxNdotV,NdotV2,TdotV2,BdotV2,ax,ax2,ay,ay2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr*irr_glsl_ggx_microfacet_to_light_measure_transform(NG_already_in_reflective_dL_measure,maxNdotL);
}

vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval(in irr_glsl_LightSample _sample, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in irr_glsl_AnisotropicMicrofacetCache _cache, in mat2x3 ior, in float ax, in float ay)
{
    if (_sample.NdotL>FLT_MIN && interaction.isotropic.NdotV>FLT_MIN)
    {
        const float TdotH2 = _cache.TdotH*_cache.TdotH;
        const float BdotH2 = _cache.BdotH*_cache.BdotH;

        const float TdotL2 = _sample.TdotL*_sample.TdotL;
        const float BdotL2 = _sample.BdotL*_sample.BdotL;

        const float TdotV2 = interaction.TdotV*interaction.TdotV;
        const float BdotV2 = interaction.BdotV*interaction.BdotV;
        return irr_glsl_ggx_height_correlated_aniso_cos_eval_wo_clamps(_cache.isotropic.NdotH2, TdotH2, BdotH2, _sample.NdotL,_sample.NdotL2,TdotL2,BdotL2, interaction.isotropic.NdotV,interaction.isotropic.NdotV_squared,TdotV2,BdotV2, _cache.isotropic.VdotH, ior, ax,ax*ax,ay,ay*ay);
    }
    else
        return vec3(0.0);
}


float irr_glsl_ggx_height_correlated_cos_eval_DG_wo_clamps(in float NdotH2, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in float a2)
{
    float NG = irr_glsl_ggx_trowbridge_reitz(a2, NdotH2);
    if (a2>FLT_MIN)
        NG *= irr_glsl_ggx_smith_correlated_wo_numerator(maxNdotV, NdotV2, maxNdotL, NdotL2, a2);

    return NG;
}

vec3 irr_glsl_ggx_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    float NG_already_in_reflective_dL_measure = irr_glsl_ggx_height_correlated_cos_eval_DG_wo_clamps(NdotH2, maxNdotL, NdotL2, maxNdotV, NdotV2, a2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);

    return fr*irr_glsl_ggx_microfacet_to_light_measure_transform(NG_already_in_reflective_dL_measure,maxNdotL);
}

vec3 irr_glsl_ggx_height_correlated_cos_eval(in irr_glsl_LightSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in irr_glsl_IsotropicMicrofacetCache _cache, in mat2x3 ior, in float a2)
{
    if (_sample.NdotL>FLT_MIN && interaction.NdotV>FLT_MIN)
        return irr_glsl_ggx_height_correlated_cos_eval_wo_clamps(_cache.NdotH2,max(_sample.NdotL,0.0),_sample.NdotL2, max(interaction.NdotV,0.0), interaction.NdotV_squared, _cache.VdotH,ior,a2);
    else
        return vec3(0.0);
}



//Heitz's 2018 paper "Sampling the GGX Distribution of Visible Normals"
vec3 irr_glsl_ggx_cos_generate(in vec3 localV, in vec2 u, in float _ax, in float _ay)
{
    vec3 V = normalize(vec3(_ax*localV.x, _ay*localV.y, localV.z));//stretch view vector so that we're sampling as if roughness=1.0

    float lensq = V.x*V.x + V.y*V.y;
    vec3 T1 = lensq > 0.0 ? vec3(-V.y, V.x, 0.0)*inversesqrt(lensq) : vec3(1.0,0.0,0.0);
    vec3 T2 = cross(V,T1);

    float r = sqrt(u.x);
    float phi = 2.0 * irr_glsl_PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + V.z);
    t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
    
    //reprojection onto hemisphere
	//TODO try it wothout the max(), not sure if -t1*t1-t2*t2>-1.0
    vec3 H = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0-t1*t1-t2*t2))*V;
    //unstretch
    return normalize(vec3(_ax*H.x, _ay*H.y, H.z));
}

// TODO: unifty the two following functions into `irr_glsl_microfacet_BRDF_cos_generate_wo_clamps(vec3 H,...)` and `irr_glsl_microfacet_BRDF_cos_generate` or at least a auto declaration macro in lieu of a template
irr_glsl_LightSample irr_glsl_ggx_cos_generate_wo_clamps(in vec3 localV, in mat3 m, in vec2 u, in float _ax, in float _ay, out irr_glsl_AnisotropicMicrofacetCache _cache)
{
    const vec3 H = irr_glsl_ggx_cos_generate(localV,u,_ax,_ay);
    
    vec3 localL;
    _cache = irr_glsl_calcAnisotropicMicrofacetCache(localV,H,localL);
    
    return irr_glsl_createLightSampleTangentSpace(localV,localL,m);
}

irr_glsl_LightSample irr_glsl_ggx_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u, in float _ax, in float _ay, out irr_glsl_AnisotropicMicrofacetCache _cache)
{
    const vec3 localV = irr_glsl_getTangentSpaceV(interaction);
    const mat3 m = irr_glsl_getTangentFrame(interaction);
    return irr_glsl_ggx_cos_generate_wo_clamps(localV,m,u,_ax,_ay,_cache);
}



float irr_glsl_ggx_pdf_wo_clamps(in float ndf, in float devsh_v, in float maxNdotV)
{
    return irr_glsl_smith_VNDF_pdf_wo_clamps(ndf,irr_glsl_GGXSmith_G1_wo_numerator(maxNdotV,devsh_v));
}
float irr_glsl_ggx_pdf_wo_clamps(in float NdotH2, in float maxNdotV, in float NdotV2, in float a2)
{
    const float ndf = irr_glsl_ggx_trowbridge_reitz(a2, NdotH2);
    const float devsh_v = irr_glsl_smith_ggx_devsh_part(NdotV2, a2, 1.0-a2);

    return irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);
}
float irr_glsl_ggx_pdf(in irr_glsl_IsotropicViewSurfaceInteraction i, irr_glsl_IsotropicMicrofacetCache h, in float a2)
{
    return irr_glsl_ggx_pdf_wo_clamps(h.NdotH2, max(i.NdotV, 0.0), i.NdotV_squared, a2);
}

float irr_glsl_ggx_pdf_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float ax, in float ay, in float ax2, in float ay2)
{
    const float ndf = irr_glsl_ggx_aniso(TdotH2,BdotH2,NdotH2, ax, ay, ax2, ay2);
    const float devsh_v = irr_glsl_smith_ggx_devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);

    return irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);
}
float irr_glsl_ggx_pdf(in irr_glsl_AnisotropicViewSurfaceInteraction i, irr_glsl_AnisotropicMicrofacetCache h, in float ax, in float ay, in float ax2, in float ay2)
{
    float TdotH2 = h.TdotH*h.TdotH;
    float BdotH2 = h.BdotH*h.BdotH;
    float maxNdotV = max(0.0,i.isotropic.NdotV);
    float NdotV2 = i.isotropic.NdotV_squared;
    float TdotV2 = i.TdotV*i.TdotV;
    float BdotV2 = i.BdotV*i.BdotV;
    return irr_glsl_ggx_pdf_wo_clamps(h.isotropic.NdotH2, TdotH2, BdotH2, maxNdotV, NdotV2, TdotV2, BdotV2, ax, ay, ax2, ay2);
}


vec3 irr_glsl_ggx_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in vec3 reflectance, in float a2)
{
    const float one_minus_a2 = 1.0 - a2;
    const float devsh_v = irr_glsl_smith_ggx_devsh_part(NdotV2, a2, one_minus_a2);
    pdf = irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);

    const float G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1_devsh(maxNdotL, NdotL2, maxNdotV, devsh_v, a2, one_minus_a2);

    return reflectance * G2_over_G1;
}

vec3 irr_glsl_ggx_cos_remainder_and_pdf(out float pdf, in irr_glsl_LightSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in irr_glsl_IsotropicMicrofacetCache _cache, in mat2x3 ior, in float a2)
{    
    if (_sample.NdotL>FLT_MIN && interaction.NdotV>FLT_MIN)
    {
        const float ndf = irr_glsl_ggx_trowbridge_reitz(a2, _cache.NdotH2);
        const vec3 reflectance = irr_glsl_fresnel_conductor(ior[0], ior[1], _cache.VdotH);

        return irr_glsl_ggx_cos_remainder_and_pdf_wo_clamps(pdf, ndf, _sample.NdotL, _sample.NdotL2, interaction.NdotV, interaction.NdotV_squared, reflectance, a2);
    }
    else
    {
        pdf = 0.0;
        return vec3(0.0);
    }
}


vec3 irr_glsl_ggx_aniso_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in vec3 reflectance, in float ax2,in float ay2)
{
    const float devsh_v = irr_glsl_smith_ggx_devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);
    pdf = irr_glsl_ggx_pdf_wo_clamps(ndf, devsh_v, maxNdotV);

    const float G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1_devsh(
        maxNdotL, TdotL2,BdotL2,NdotL2,
        maxNdotV, devsh_v,
        ax2, ay2
    );

    return reflectance * G2_over_G1;
}

vec3 irr_glsl_ggx_aniso_cos_remainder_and_pdf(out float pdf, in irr_glsl_LightSample _sample, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in irr_glsl_AnisotropicMicrofacetCache _cache, in mat2x3 ior, in float ax, in float ay)
{
    if (_sample.NdotL>FLT_MIN && interaction.isotropic.NdotV>FLT_MIN)
    {
        const float TdotH2 = _cache.TdotH*_cache.TdotH;
        const float BdotH2 = _cache.BdotH*_cache.BdotH;

        const float TdotL2 = _sample.TdotL*_sample.TdotL;
        const float BdotL2 = _sample.BdotL*_sample.BdotL;

        const float TdotV2 = interaction.TdotV*interaction.TdotV;
        const float BdotV2 = interaction.BdotV*interaction.BdotV;

        const float ax2 = ax*ax;
        const float ay2 = ay*ay;
        const float ndf = irr_glsl_ggx_aniso(TdotH2,BdotH2,_cache.isotropic.NdotH2, ax, ay, ax2, ay2);
        const vec3 reflectance = irr_glsl_fresnel_conductor(ior[0], ior[1], _cache.isotropic.VdotH);

	    return irr_glsl_ggx_aniso_cos_remainder_and_pdf_wo_clamps(pdf, ndf, _sample.NdotL, _sample.NdotL2, TdotL2, BdotL2, interaction.isotropic.NdotV, TdotV2, BdotV2, interaction.isotropic.NdotV_squared, reflectance, ax2, ay2);
    }
    else
    {
        pdf = 0.0;
        return vec3(0.0);
    }
}

#endif
