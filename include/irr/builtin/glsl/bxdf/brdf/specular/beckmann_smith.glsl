#ifndef _IRR_BSDF_BRDF_SPECULAR_BECKMANN_SMITH_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_BECKMANN_SMITH_INCLUDED_
//TODO rename this file to beckmann.glsl
#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ndf/beckmann.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/geom/smith.glsl>
#include <irr/builtin/glsl/math/functions.glsl>

#include <irr/builtin/glsl/math/functions.glsl>

irr_glsl_BSDFSample irr_glsl_beckmann_smith_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample, in float ax, in float ay)
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

	return irr_glsl_createBSDFSample(H,localV,dot(H,localV),m);
}

vec3 irr_glsl_beckmann_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float a2)
{
	float NdotL2 = s.NdotL*s.NdotL;
	float lambda_V = irr_glsl_smith_beckmann_Lambda(interaction.NdotV_squared, a2);
	float onePlusLambda_V = 1.0 + lambda_V;

	float G1 = 1.0 / onePlusLambda_V;
	pdf = irr_glsl_beckmann(a2,s.NdotH*s.NdotH)*G1*0.25/interaction.NdotV;

	float G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(onePlusLambda_V, s.NdotL, NdotL2, a2);
	
	vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], s.VdotH);
	return fr*G2_over_G1;
}
vec3 irr_glsl_beckmann_aniso_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
	float NdotL2 = s.NdotL*s.NdotL;
    float TdotV2 = interaction.TdotV*interaction.TdotV;
    float BdotV2 = interaction.BdotV*interaction.BdotV;

    float c2 = irr_glsl_smith_beckmann_C2(TdotV2, BdotV2, interaction.isotropic.NdotV_squared, ax2, ay2);
	float lambda_V = irr_glsl_smith_beckmann_Lambda(c2);
	float onePlusLambda_V = 1.0 + lambda_V;

	float G1 = 1.0 / onePlusLambda_V;
	pdf = irr_glsl_beckmann(ax,ay,ax2,ay2,s.TdotH*s.TdotH,s.BdotH*s.BdotH,s.NdotH*s.NdotH)*G1*0.25/interaction.isotropic.NdotV;

	float G2_over_G1 = irr_glsl_beckmann_smith_G2_over_G1(onePlusLambda_V, s.NdotL, s.TdotL*s.TdotL, s.BdotL*s.BdotL, NdotL2, ax2, ay2);
	
	vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], s.VdotH);
	return fr*G2_over_G1;
}

float irr_glsl_beckmann_smith_height_correlated_cos_eval_DG(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in float a2)
{
    float ndf = irr_glsl_beckmann(a2, params.NdotH*params.NdotH);
    float scalar_part = ndf / (4.0 * interaction.NdotV);
    if  (a2>FLT_MIN)
    {
        float g = irr_glsl_beckmann_smith_correlated(interaction.NdotV_squared, params.NdotL_squared, a2);
        scalar_part *= g;
    }
    
    return scalar_part;
}
vec3 irr_glsl_beckmann_smith_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction interaction,  in mat2x3 ior, in float a2)
{
    float scalar_part = irr_glsl_beckmann_smith_height_correlated_cos_eval_DG(params, interaction, a2);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], params.VdotH);
    
    return scalar_part*fr;
}

float irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_DG(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in float ax, in float ax2, in float ay, in float ay2)
{
    float ndf = irr_glsl_beckmann(ax, ay, ax2, ay2, params.TdotH*params.TdotH, params.BdotH*params.BdotH, params.isotropic.NdotH*params.isotropic.NdotH);
    float scalar_part = ndf / (4.0 * interaction.isotropic.NdotV);
    if (ax>FLT_MIN || ay>FLT_MIN)
    {
        float TdotV2 = interaction.TdotV*interaction.TdotV;
        float BdotV2 = interaction.BdotV*interaction.BdotV;
        float TdotL2 = params.TdotL*params.TdotL;
        float BdotL2 = params.BdotL*params.BdotL;
        float g = irr_glsl_beckmann_smith_correlated(TdotV2, BdotV2, interaction.isotropic.NdotV_squared, TdotL2, BdotL2, params.isotropic.NdotL_squared, ax2, ay2);
        scalar_part *= g;
    }
    
    return scalar_part;
}
vec3 irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction interaction,  in mat2x3 ior, in float ax, in float ax2, in float ay, in float ay2)
{
    float scalar_part = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval_DG(params, interaction, ax, ax2, ay, ay2);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], params.isotropic.VdotH);
    
    return scalar_part*fr;
}

#endif
