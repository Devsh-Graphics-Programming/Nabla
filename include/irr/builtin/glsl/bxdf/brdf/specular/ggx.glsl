#ifndef _IRR_BSDF_BRDF_SPECULAR_GGX_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_GGX_INCLUDED_

#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ndf/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/geom/smith.glsl>

vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in irr_glsl_AnisotropicViewSurfaceInteraction inter, in mat2x3 ior, in float a2, in vec2 atb, in float aniso)
{
    float g = irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(atb.x, atb.y, params.TdotL, inter.TdotV, params.BdotL, inter.BdotV, params.isotropic.NdotL, inter.isotropic.NdotV);
    float ndf = irr_glsl_ggx_burley_aniso(aniso, a2, params.TdotH, params.BdotH, params.isotropic.NdotH);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], params.isotropic.VdotH);

    return params.isotropic.NdotL * g*ndf*fr;
}
vec3 irr_glsl_ggx_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in mat2x3 ior, in float a2)
{
    float g = irr_glsl_ggx_smith_height_correlated_wo_numerator(a2, params.NdotL, inter.NdotV);
    float ndf = irr_glsl_ggx_trowbridge_reitz(a2, params.NdotH*params.NdotH);
    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], params.VdotH);

    return params.NdotL * g*ndf*fr;
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
irr_glsl_BSDFSample irr_glsl_ggx_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 _sample, in float _ax, in float _ay)
{
    vec2 u = vec2(_sample)/float(UINT_MAX);
    return irr_glsl_ggx_cos_generate(interaction, u, _ax, _ay);
}

vec3 irr_glsl_ggx_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float a2)
{
	float one_minus_a2 = 1.0-a2;
	float G1 = irr_glsl_GGXSmith_G1_(interaction.NdotV,a2,one_minus_a2);
    float absVdotH = abs(s.VdotH);
	pdf = irr_glsl_ggx_trowbridge_reitz(a2,s.NdotH*s.NdotH)*G1*absVdotH/interaction.NdotV;
	
	float devsh_v = irr_glsl_smith_ggx_devsh_part(interaction.NdotV_squared,a2,one_minus_a2);
	float G2_over_G1 = s.NdotL*(devsh_v + interaction.NdotV);
	G2_over_G1 /= interaction.NdotV*irr_glsl_smith_ggx_devsh_part(s.NdotL*s.NdotL,a2,one_minus_a2) + s.NdotL*devsh_v;
	
	vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], s.VdotH);
	return fr*G2_over_G1*interaction.NdotV/absVdotH;
}

vec3 irr_glsl_ggx_aniso_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in mat2x3 ior, in float ax, in float ay)
{
    float ax2 = ax*ax;
    float ay2 = ay*ay;
    float a02_v = irr_glsl_ggx_smith_aniso_a0_2(interaction.isotropic.N, interaction.isotropic.V.dir, interaction.T, interaction.isotropic.NdotV_squared, ax2, ay2);
    float one_minus_a02_v = 1.0 - a02_v;
    float G1 = irr_glsl_GGXSmith_G1_(interaction.isotropic.NdotV,a02_v,one_minus_a02_v);
    float absVdotH = abs(s.VdotH);
    pdf = irr_glsl_ggx_aniso(s.TdotH*s.TdotH,s.BdotH*s.BdotH,s.NdotH*s.NdotH,ax,ay,ax2,ay2)*G1*absVdotH/interaction.isotropic.NdotV;

    float a02_l = irr_glsl_ggx_smith_aniso_a0_2(interaction.isotropic.N, s.L, interaction.T, s.NdotL*s.NdotL, ax2, ay2);
    float devsh_v = irr_glsl_smith_ggx_devsh_part(interaction.NdotV_squared,a02_v,one_minus_a02_v);
    float G2_over_G1 = s.NdotL*(devsh_v + interaction.isotropic.NdotV);
    G2_over_G1 /= interaction.isotropic.NdotV*irr_glsl_smith_ggx_devsh_part(s.NdotL*s.NdotL,a02_l,1.0-a02_l) + s.NdotL*devsh_v;

	vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], s.VdotH);
	return fr*G2_over_G1*interaction.NdotV/absVdotH;
}

#endif
