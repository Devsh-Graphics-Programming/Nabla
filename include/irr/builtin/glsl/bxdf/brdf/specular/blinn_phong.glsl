#ifndef _IRR_BSDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_BLINN_PHONG_INCLUDED_

#include <irr/builtin/glsl/bxdf/common.glsl>
#include <irr/builtin/glsl/bxdf/common_samples.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ndf/blinn_phong.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/geom/smith.glsl>

//https://zhuanlan.zhihu.com/p/58205525
irr_glsl_BSDFSample irr_glsl_blinn_phong_cos_generate(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample, in float n)
{
    vec2 u = _sample;

    mat3 m = irr_glsl_getTangentFrame(interaction);

    float phi = 2.0*irr_glsl_PI*u.y;
    float cosTheta = pow(u.x, 1.0/(n+1.0));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    vec3 H = vec3(cosPhi*sinTheta, sinPhi*sinTheta, cosTheta);
    vec3 localV = interaction.isotropic.V.dir*m;

	return irr_glsl_createBSDFSample(H,localV,dot(H,localV),m);
}

//TODO remainder and pdf
vec3 irr_glsl_blinn_phong_dielectric_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in float n, in vec3 ior)
{
	pdf = (n+1.0)*0.5*irr_glsl_RECIPROCAL_PI * 0.25*pow(s.NdotH,n)/s.VdotH;

    vec3 fr = irr_glsl_fresnel_dielectric(ior, s.VdotH);
    return fr * s.NdotL * (n*(n + 6.0) + 8.0) * s.VdotH / ((pow(0.5,0.5*n) + n) * (n + 1.0));
}

vec3 irr_glsl_blinn_phong_conductor_cos_remainder_and_pdf(out float pdf, in irr_glsl_BSDFSample s, in irr_glsl_IsotropicViewSurfaceInteraction interaction, in float n, in mat2x3 ior)
{
	pdf = (n+1.0)*0.5*irr_glsl_RECIPROCAL_PI * 0.25*pow(s.NdotH,n)/s.VdotH;

    vec3 fr = irr_glsl_fresnel_conductor(ior[0], ior[1], s.VdotH);
    return fr * s.NdotL * (n*(n + 6.0) + 8.0) * s.VdotH / ((pow(0.5,0.5*n) + n) * (n + 1.0));
}

vec3 irr_glsl_blinn_phong_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in float n, in mat2x3 ior, in float a2)
{
    float g = irr_glsl_beckmann_smith_correlated(inter.NdotV_squared, params.NdotL_squared, a2);
    float d = irr_glsl_blinn_phong(params.NdotH, n);
    return g*d*irr_glsl_fresnel_conductor(ior[0], ior[1], params.VdotH) / (4.0*inter.NdotV);
}
vec3 irr_glsl_blinn_phong_cos_eval(in irr_glsl_BSDFIsotropicParams params, in irr_glsl_IsotropicViewSurfaceInteraction inter, in float n, in mat2x3 ior)
{
    float a = sqrt(2.0/(n+2.0));
    return irr_glsl_blinn_phong_cos_eval(params, inter, n, ior, a*a);
}

#endif
