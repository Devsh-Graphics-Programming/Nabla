#ifndef _IRR_BUILTIN_GLSL_BXDF_BRDF_SPECULAR_BECKMANN_INCLUDED_
#define _IRR_BUILTIN_GLSL_BXDF_BRDF_SPECULAR_BECKMANN_INCLUDED_

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/fresnel.glsl>
#include <irr/builtin/glsl/bxdf/ndf/beckmann.glsl>
#include <irr/builtin/glsl/bxdf/geom/smith/beckmann.glsl>
#include <irr/builtin/glsl/math/functions.glsl>

#include <irr/builtin/glsl/math/functions.glsl>

irr_glsl_BxDFSample irr_glsl_beckmann_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample, in float ax, in float ay)
{
    vec2 u = _sample;
    
    mat3 m = irr_glsl_getTangentFrame(interaction);

    vec3 localV = interaction.isotropic.V.dir;
    localV = normalize(localV*m);//transform to tangent space
    //stretch
    vec3 V = normalize(vec3(ax*localV.x, ay*localV.y, localV.z));

    vec2 slope;
    if (V.z > 0.9999)//V.z=NdotV=cosTheta in tangent space
    {
        float r = sqrt(-log(1.0-u.x));
        float sinPhi = sin(2.0*irr_glsl_PI*u.y);
        float cosPhi = cos(2.0*irr_glsl_PI*u.y);
        slope = vec2(r)*vec2(cosPhi,sinPhi);
    }
    else
    {
        float cosTheta = V.z;
        float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
        float tanTheta = sinTheta/cosTheta;
        float cotTheta = 1.0/tanTheta;
        
        float a = -1.0;
        float c = irr_glsl_erf(cosTheta);
        float sample_x = max(u.x, 1.0e-6);
        float theta = acos(cosTheta);
        float fit = 1.0 + theta * (-0.876 + theta * (0.4265 - 0.0594*theta));
        float b = c - (1.0 + c) * pow(1.0-sample_x, fit);
        
        float normalization = 1.0 / (1.0 + c + irr_glsl_SQRT_RECIPROCAL_PI * tanTheta * exp(-cosTheta*cosTheta));

        const int ITER_THRESHOLD = 10;
		const float MAX_ACCEPTABLE_ERR = 1.0e-5;
        int it = 0;
        float value=1000.0;
        while (++it<ITER_THRESHOLD && abs(value)>MAX_ACCEPTABLE_ERR)
        {
            if (!(b>=a && b<=c))
                b = 0.5 * (a+c);

            float invErf = irr_glsl_erfInv(b);
            value = normalization * (1.0 + b + irr_glsl_SQRT_RECIPROCAL_PI * tanTheta * exp(-invErf*invErf)) - sample_x;
            float derivative = normalization * (1.0 - invErf*cosTheta);

            if (value > 0.0)
                c = b;
            else
                a = b;

            b -= value/derivative;
        }
        slope.x = irr_glsl_erfInv(b);
        slope.y = irr_glsl_erfInv(2.0 * max(u.y,1.0e-6) - 1.0);
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

    vec3 H = normalize(vec3(-slope, 1.0));

	return irr_glsl_createBRDFSample(H,localV,dot(H,localV),m);
}



float irr_glsl_beckmann_pdf_wo_clamps(in float ndf, in float lambdaV, in float maxNdotV)
{
    float G1 = 1.0 / (1.0 + lambdaV);

    return ndf*G1*0.25 / maxNdotV;
}

float irr_glsl_beckmann_pdf_wo_clamps(in float NdotH2, in float maxNdotV, in float NdotV2, in float a2)
{
    float lambda = irr_glsl_smith_beckmann_Lambda(NdotV2, a2);
    float ndf = irr_glsl_beckmann(a2, NdotH2);

    return irr_glsl_beckmann_pdf_wo_clamps(ndf, lambda, maxNdotV);
}

float irr_glsl_beckmann_pdf_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float ax, in float ax2, in float ay, in float ay2)
{
    float c2 = irr_glsl_smith_beckmann_C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
    float lambda = irr_glsl_smith_beckmann_Lambda(c2);
    float ndf = irr_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);

    return irr_glsl_beckmann_pdf_wo_clamps(ndf, lambda, maxNdotV);
}



vec3 irr_glsl_beckmann_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float maxNdotL, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    float lambda_V = irr_glsl_smith_beckmann_Lambda(NdotV2, a2);
    float onePlusLambda_V = 1.0 + lambda_V;

    float G1 = 1.0 / onePlusLambda_V;
    pdf = ndf * G1 * 0.25 / maxNdotV;

    float G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(onePlusLambda_V, maxNdotL, NdotL2, a2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr * G2_over_G1;
}
vec3 irr_glsl_beckmann_cos_remainder_and_pdf(out float pdf, in irr_glsl_BxDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float a2)
{
    const float NdotH2 = s.NdotH * s.NdotH;

    const float NdotL2 = s.NdotL * s.NdotL;

    const float ndf = irr_glsl_beckmann(a2, NdotH2);
	
    return irr_glsl_beckmann_cos_remainder_and_pdf_wo_clamps(pdf, ndf, max(s.NdotL,0.0), NdotL2, max(interaction.NdotV,0.0), interaction.NdotV_squared, s.VdotH, ior, a2);
}



vec3 irr_glsl_beckmann_aniso_cos_remainder_and_pdf_wo_clamps(out float pdf, in float ndf, in float maxNdotL, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float TdotV2, in float BdotV2, in float NdotV2, in float VdotH, in mat2x3 ior, in float ax2, in float ay2)
{
    float c2 = irr_glsl_smith_beckmann_C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
    float lambda_V = irr_glsl_smith_beckmann_Lambda(c2);
    float onePlusLambda_V = 1.0 + lambda_V;

    float G1 = 1.0 / onePlusLambda_V;
    pdf = ndf * G1 * 0.25 / maxNdotV;

    float G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(onePlusLambda_V, maxNdotL, TdotL2, BdotL2, NdotL2, ax2, ay2);

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    return fr * G2_over_G1;
}
vec3 irr_glsl_beckmann_aniso_cos_remainder_and_pdf(out float pdf, in irr_glsl_BxDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    const float NdotH2 = s.NdotH * s.NdotH;
    const float TdotH2 = s.TdotH * s.TdotH;
    const float BdotH2 = s.BdotH * s.BdotH;

    const float NdotL2 = s.NdotL * s.NdotL;
    const float TdotL2 = s.TdotL * s.TdotL;
    const float BdotL2 = s.BdotL * s.BdotL;
    
    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;
    
    const float ndf = irr_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);

	return irr_glsl_beckmann_aniso_cos_remainder_and_pdf_wo_clamps(pdf, ndf, max(s.NdotL,0.0), NdotL2,TdotL2,BdotL2, max(interaction.isotropic.NdotV,0.0),TdotV2,BdotV2, interaction.isotropic.NdotV_squared, s.VdotH, ior, ax2, ay2);
}



float irr_glsl_beckmann_height_correlated_cos_eval_DG_wo_clamps(in float NdotH2, in float NdotL2, in float maxNdotV, in float NdotV2, in float a2)
{
    float ndf = irr_glsl_beckmann(a2, NdotH2);
    float scalar_part = ndf / (4.0 * maxNdotV);
    if  (a2>FLT_MIN)
    {
        float g = irr_glsl_beckmann_smith_correlated(NdotV2, NdotL2, a2);
        scalar_part *= g;
    }
    
    return scalar_part;
}
vec3 irr_glsl_beckmann_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float NdotL2, in float maxNdotV, in float NdotV2, in float VdotH, in mat2x3 ior, in float a2)
{
    float scalar_part = irr_glsl_beckmann_height_correlated_cos_eval_DG_wo_clamps(NdotH2, NdotL2, maxNdotV, NdotV2, a2);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    
    return scalar_part*fr;
}
vec3 irr_glsl_beckmann_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float a2)
{
    const float NdotH2 = params.NdotH * params.NdotH;

    return irr_glsl_beckmann_height_correlated_cos_eval_wo_clamps(NdotH2,params.NdotL_squared,max(interaction.NdotV,0.0),interaction.NdotV_squared,params.VdotH,ior,a2);
}

float irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float ax, in float ax2, in float ay, in float ay2)
{
    float ndf = irr_glsl_beckmann(ax, ay, ax2, ay2, TdotH2, BdotH2, NdotH2);
    float scalar_part = ndf / (4.0 * maxNdotV);
    if (ax>FLT_MIN || ay>FLT_MIN)
    {
        float g = irr_glsl_beckmann_smith_correlated(TdotV2, BdotV2, NdotV2, TdotL2, BdotL2, NdotL2, ax2, ay2);
        scalar_part *= g;
    }
    
    return scalar_part;
}
vec3 irr_glsl_beckmann_aniso_height_correlated_cos_eval_wo_clamps(in float NdotH2, in float TdotH2, in float BdotH2, in float NdotL2, in float TdotL2, in float BdotL2, in float maxNdotV, in float NdotV2, in float TdotV2, in float BdotV2, in float VdotH, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    float scalar_part = irr_glsl_beckmann_aniso_height_correlated_cos_eval_DG_wo_clamps(NdotH2,TdotH2,BdotH2, NdotL2,TdotL2,BdotL2, maxNdotV,NdotV2,TdotV2,BdotV2, ax, ax2, ay, ay2);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], VdotH);
    
    return scalar_part*fr;
}
vec3 irr_glsl_beckmann_aniso_height_correlated_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    const float NdotH2 = params.isotropic.NdotH * params.isotropic.NdotH;
    const float TdotH2 = params.TdotH * params.TdotH;
    const float BdotH2 = params.BdotH * params.BdotH;

    const float TdotL2 = params.TdotL * params.TdotL;
    const float BdotL2 = params.BdotL * params.BdotL;

    const float TdotV2 = interaction.TdotV * interaction.TdotV;
    const float BdotV2 = interaction.BdotV * interaction.BdotV;

    return irr_glsl_beckmann_aniso_height_correlated_cos_eval_wo_clamps(NdotH2,TdotH2,BdotH2, params.isotropic.NdotL_squared,TdotL2,BdotL2, max(interaction.isotropic.NdotV,0.0),interaction.isotropic.NdotV_squared,TdotV2,BdotV2, params.isotropic.VdotH, ior,ax,ax2,ay,ay2);
}

#endif
