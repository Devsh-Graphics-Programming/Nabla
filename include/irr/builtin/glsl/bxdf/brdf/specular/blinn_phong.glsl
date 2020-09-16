#ifndef _IRR_BUILTIN_GLSL_BXDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_

#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/fresnel.glsl>
#include <irr/builtin/glsl/bxdf/ndf/blinn_phong.glsl>
#include <irr/builtin/glsl/bxdf/geom/smith/beckmann.glsl>

//conversion between alpha and Phong exponent, Walter et.al.
float irr_glsl_phong_exp_to_alpha2(in float n)
{
    return 2.0/(n+2.0);
}
//+INF for a2==0.0
float irr_glsl_alpha2_to_phong_exp(in float a2)
{
    return 2.0/a2 - 2.0;
}

//https://zhuanlan.zhihu.com/p/58205525
//only NDF sampling
//however we dont really care about phong sampling
vec3 irr_glsl_blinn_phong_cos_generate(in vec2 u, in float n)
{
    float phi = 2.0*irr_glsl_PI*u.y;
    float cosTheta = pow(u.x, 1.0/(n+1.0));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    return vec3(cosPhi*sinTheta, sinPhi*sinTheta, cosTheta);
}
irr_glsl_BxDFSample irr_glsl_blinn_phong_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 u, in float n)
{
    const vec3 localV = irr_glsl_getTangentSpaceV(interaction);
    const vec3 H = irr_glsl_blinn_phong_cos_generate(u,ax,ay);
    const mat3 m = irr_glsl_getTangentFrame(interaction);
    return irr_glsl_createBRDFSample(H, localV, dot(H,localV), m);
}

/*
vec3 irr_glsl_blinn_phong_dielectric_cos_remainder_and_pdf(out float pdf, in irr_glsl_BxDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in float n, in vec3 ior)
{
	pdf = (n+1.0)*0.5*irr_glsl_RECIPROCAL_PI * 0.25*pow(s.NdotH,n)/s.VdotH;

    vec3 fr = irr_glsl_fresnel_dielectric(ior, s.VdotH);
    return fr * s.NdotL * (n*(n + 6.0) + 8.0) * s.VdotH / ((pow(0.5,0.5*n) + n) * (n + 1.0));
}

vec3 irr_glsl_blinn_phong_conductor_cos_remainder_and_pdf(out float pdf, in irr_glsl_BxDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in float n, in mat2x3 ior)
{
	pdf = (n+1.0)*0.5*irr_glsl_RECIPROCAL_PI * 0.25*pow(s.NdotH,n)/s.VdotH;

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], s.VdotH);
    return fr * s.NdotL * (n*(n + 6.0) + 8.0) * s.VdotH / ((pow(0.5,0.5*n) + n) * (n + 1.0));
}
*/

float irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(in float NdotH, in float maxNdotV, in float NdotV_squared, in float NdotL_squared, in float n, in float a2)
{
    float d = irr_glsl_blinn_phong(NdotH, n);
    float scalar_part = d/(4.0*maxNdotV);
    if (a2>FLT_MIN)
    {
        float g = irr_glsl_beckmann_smith_correlated(NdotV_squared, NdotL_squared, a2);
        scalar_part *= g;
    }
    return scalar_part;
}
float irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(in float NdotH, in float maxNdotV, in float NdotV_squared, in float NdotL_squared, in float n)
{
    float a2 = irr_glsl_phong_exp_to_alpha2(n);
    return irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(NdotH, maxNdotV, NdotV_squared, NdotL_squared, n, a2);
}

vec3 irr_glsl_blinn_phong_cos_eval_wo_clamps(in float NdotH, in float maxNdotV, in float NdotV_squared, in float NdotL_squared, in float VdotH, in float n, in mat2x3 ior, in float a2)
{
    float scalar_part = irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(NdotH, maxNdotV, NdotV_squared, NdotL_squared, n, a2);
    return scalar_part*irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
}
vec3 irr_glsl_blinn_phong_cos_eval_wo_clamps(in float NdotH, in float maxNdotV, in float NdotV_squared, in float NdotL_squared, in float VdotH, in float n, in mat2x3 ior)
{
    float a2 = irr_glsl_phong_exp_to_alpha2(n);
    return irr_glsl_blinn_phong_cos_eval_wo_clamps(NdotH, maxNdotV, NdotV_squared, NdotL_squared, VdotH, n, ior, a2);
}
vec3 irr_glsl_blinn_phong_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in float n, in mat2x3 ior)
{
    const float maxNdotV = max(inter.NdotV,0.0);
    return irr_glsl_blinn_phong_cos_eval_wo_clamps(params.NdotH, maxNdotV, inter.NdotV_squared, params.NdotL_squared, params.VdotH, n, ior);
}


float irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(in float NdotH, in float NdotH2, in float TdotH2, in float BdotH2, float TdotL2, float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV_squared, in float NdotL_squared, in float nx, in float ny, in float ax2, in float ay2)
{
    float d = irr_glsl_blinn_phong(NdotH, 1.0/(1.0-NdotH2), TdotH2, BdotH2, nx, ny);
    float scalar_part = d/(4.0*maxNdotV);
    if (ax2>FLT_MIN || ay2>FLT_MIN)
    {
        float g = irr_glsl_beckmann_smith_correlated(TdotV2, BdotV2, NdotV_squared, TdotL2, BdotL2, NdotL_squared, ax2, ay2);
        scalar_part *= g;
    }

    return scalar_part;
}
float irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(in float NdotH, in float NdotH2, in float TdotH2, in float BdotH2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV_squared, in float NdotL_squared, in float nx, in float ny)
{
    float ax2 = irr_glsl_phong_exp_to_alpha2(nx);
    float ay2 = irr_glsl_phong_exp_to_alpha2(ny);

    return irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotH2, TdotH2, BdotH2, TdotL2, BdotL2, maxNdotV, TdotV2, BdotV2, NdotV_squared, NdotL_squared, nx, ny, ax2, ay2);
}

vec3 irr_glsl_blinn_phong_cos_eval_wo_clamps(in float NdotH, in float NdotH2, in float TdotH2, in float BdotH2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV_squared, in float NdotL_squared, in float VdotH, in float nx, in float ny, in mat2x3 ior, in float ax2, in float ay2)
{
    float scalar_part = irr_glsl_blinn_phong_cos_eval_DG_wo_clamps(NdotH, NdotH2, TdotH2, BdotH2, TdotL2, BdotL2, maxNdotV, TdotV2, BdotV2, NdotV_squared, NdotL_squared, nx, ny, ax2, ay2);

    return scalar_part*irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
}
vec3 irr_glsl_blinn_phong_cos_eval_wo_clamps(in float NdotH, in float NdotH2, in float TdotH2, in float BdotH2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV_squared, in float NdotL_squared, in float VdotH, in float nx, in float ny, in mat2x3 ior)
{
    float ax2 = irr_glsl_phong_exp_to_alpha2(nx);
    float ay2 = irr_glsl_phong_exp_to_alpha2(ny);

    return irr_glsl_blinn_phong_cos_eval_wo_clamps(NdotH, NdotH2, TdotH2, BdotH2, TdotL2, BdotL2, maxNdotV, TdotV2, BdotV2, NdotV_squared, NdotL_squared, VdotH, nx, ny, ior, ax2, ay2);
}
vec3 irr_glsl_blinn_phong_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction inter, in float nx, in float ny, in mat2x3 ior)
{
    const float NdotH2 = params.isotropic.NdotH * params.isotropic.NdotH;
    const float TdotH2 = params.TdotH * params.TdotH;
    const float BdotH2 = params.BdotH * params.BdotH;
    const float TdotL2 = params.TdotL * params.TdotL;
    const float BdotL2 = params.BdotL * params.BdotL;
    const float maxNdotV = max(inter.isotropic.NdotV, 0.0);
    const float TdotV2 = inter.TdotV * inter.TdotV;
    const float BdotV2 = inter.BdotV * inter.BdotV;
    return irr_glsl_blinn_phong_cos_eval_wo_clamps(params.isotropic.NdotH, NdotH2, TdotH2, BdotH2, TdotL2, BdotL2, maxNdotV, TdotV2, BdotV2, inter.isotropic.NdotV_squared, params.isotropic.NdotL_squared, params.isotropic.VdotH, nx, ny, ior);
}
#endif
