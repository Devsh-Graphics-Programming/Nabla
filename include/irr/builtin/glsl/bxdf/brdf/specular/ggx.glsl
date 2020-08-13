#ifndef _IRR_BSDF_BRDF_SPECULAR_GGX_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_GGX_INCLUDED_

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ndf/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/geom/smith.glsl>

//depr
/*
vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction inter, in mat2x3 ior, in float a2, in vec2 atb, in float aniso)
{
    float g = irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(atb.x, atb.y, params.TdotL, inter.TdotV, params.BdotL, inter.BdotV, params.isotropic.NdotL, inter.isotropic.NdotV);
    float ndf = irr_glsl_ggx_burley_aniso(aniso, a2, params.TdotH, params.BdotH, params.isotropic.NdotH);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], params.isotropic.VdotH);

    return params.isotropic.NdotL * g*ndf*fr;
}
*/
//defined using NDF function with better API (compared to burley used above) and new impl of correlated smith
vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction inter, in mat2x3 ior, in float ax, in float ay)
{
    float ax2 = ax*ax;
    float ay2 = ay*ay;
    float ndf = irr_glsl_ggx_aniso(params.TdotH*params.TdotH, params.BdotH*params.BdotH, params.isotropic.NdotH*params.isotropic.NdotH, ax, ay, ax2, ay2);
    float scalar_part = ndf*params.isotropic.NdotL;
    if (ax>FLT_MIN || ay>FLT_MIN)
    {
        float g = irr_glsl_ggx_smith_correlated_wo_numerator(
            inter.isotropic.NdotV, inter.TdotV * inter.TdotV, inter.BdotV * inter.BdotV, inter.isotropic.NdotV_squared,
            params.isotropic.NdotL, params.TdotL * params.TdotL, params.BdotL * params.BdotL, params.isotropic.NdotL_squared,
            ax2, ay2
        );
        scalar_part *= g;
    }


    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], params.isotropic.VdotH);
    return fr*scalar_part;
}
vec3 irr_glsl_ggx_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in mat2x3 ior, in float a2)
{
    float ndf = irr_glsl_ggx_trowbridge_reitz(a2, params.NdotH*params.NdotH);
    float scalar_part = ndf*params.NdotL;
    if (a2>FLT_MIN)
    {
        float g = irr_glsl_ggx_smith_correlated_wo_numerator(inter.NdotV, inter.NdotV_squared, params.NdotL, params.NdotL_squared, a2);
        scalar_part *= g;
    }
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], params.VdotH);
    return fr*scalar_part;
}

//Heitz's 2018 paper "Sampling the GGX Distribution of Visible Normals"
irr_glsl_BSDFSample irr_glsl_ggx_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample, in float _ax, in float _ay)
{
    vec2 u = _sample;

    mat3 m = irr_glsl_getTangentFrame(interaction);

    vec3 localV = interaction.isotropic.V.dir;
    localV = normalize(localV*m);//transform to tangent space
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
    vec3 H = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0-t1*t1-t2*t2))*V;
    //unstretch
    H = normalize(vec3(_ax*H.x, _ay*H.y, max(0.0,H.z)));
    float NdotH = H.z;

	return irr_glsl_createBSDFSample(H,localV,dot(H,localV),m);
}

vec3 irr_glsl_ggx_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float a2)
{
	float one_minus_a2 = 1.0-a2;
	pdf = irr_glsl_ggx_trowbridge_reitz(a2,s.NdotH*s.NdotH)*irr_glsl_GGXSmith_G1_wo_numerator(interaction.NdotV,a2,one_minus_a2)*0.5;
	
    float G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1(s.NdotL, s.NdotL*s.NdotL, interaction.NdotV, interaction.NdotV_squared, a2, one_minus_a2);
	
	vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], s.VdotH);
	return fr*G2_over_G1;
}

vec3 irr_glsl_ggx_aniso_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float ax, in float ay)
{
    float ax2 = ax*ax;
    float ay2 = ay*ay;
    float TdotV2 = interaction.TdotV*interaction.TdotV;
    float BdotV2 = interaction.BdotV*interaction.BdotV;
    pdf = irr_glsl_ggx_aniso(s.TdotH*s.TdotH,s.BdotH*s.BdotH,s.NdotH*s.NdotH,ax,ay,ax2,ay2)*irr_glsl_GGXSmith_G1_wo_numerator(interaction.isotropic.NdotV, TdotV2, BdotV2, interaction.isotropic.NdotV_squared, ax2, ay2)*0.5;

    float G2_over_G1 = irr_glsl_ggx_smith_G2_over_G1(
        s.NdotL, s.TdotL*s.TdotL, s.BdotL*s.BdotL, s.NdotL*s.NdotL,
        interaction.isotropic.NdotV, TdotV2, BdotV2, interaction.isotropic.NdotV_squared,
        ax2, ay2
    );

	vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], s.VdotH);
	return fr*G2_over_G1;
}

#endif
