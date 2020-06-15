#ifndef _IRR_BSDF_BRDF_SPECULAR_GGX_INCLUDED_
#define _IRR_BSDF_BRDF_SPECULAR_GGX_INCLUDED_

#include <irr/builtin/glsl/bsdf/common.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/ndf/ggx.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/geom/smith.glsl>
#include <irr/builtin/glsl/bsdf/brdf/specular/fresnel/fresnel.glsl>

vec3 irr_glsl_ggx_height_correlated_aniso_cos_eval(in irr_glsl_BSDFAnisotropicParams params, in mat2x3 ior2, in float a2, in vec2 atb, in float aniso)
{
    float g = irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(atb.x, atb.y, params.TdotL, params.TdotV, params.BdotL, params.BdotV, params.isotropic.NdotL, params.isotropic.interaction.NdotV);
    float ndf = irr_glsl_ggx_burley_aniso(aniso, a2, params.TdotH, params.BdotH, params.isotropic.NdotH);
    vec3 fr = irr_glsl_fresnel_conductor(ior2[0], ior2[1], params.isotropic.VdotH);

    return params.isotropic.NdotL * g*ndf*fr;
}
vec3 irr_glsl_ggx_height_correlated_cos_eval(in irr_glsl_BSDFIsotropicParams params, in mat2x3 ior2, in float a2)
{
    float g = irr_glsl_ggx_smith_height_correlated_wo_numerator(a2, params.NdotL, params.interaction.NdotV);
    float ndf = irr_glsl_ggx_trowbridge_reitz(a2, params.NdotH*params.NdotH);
    vec3 fr = irr_glsl_fresnel_conductor(ior2[0], ior2[1], params.VdotH);

    return params.NdotL * g*ndf*fr;
}

//Heitz's 2018 paper "Sampling the GGX Distribution of Visible Normals"
//Also: problem is our anisotropic ggx ndf (above) has extremely weird API (anisotropy and a2 instead of ax and ay) and so it's incosistent with sampling function
//  currently using isotropic trowbridge_reitz for PDF
irr_glsl_BSDFSample irr_glsl_ggx_smith_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in vec2 _sample, in float _ax, in float _ay)
{
    vec2 u = _sample;

    mat3 m = irr_glsl_getTangentFrame(interaction);

    vec3 V = interaction.isotropic.V.dir;
    V = normalize(V*m);//transform to tangent space
    V = normalize(vec3(_ax*V.x, _ay*V.y, V.z));//stretch view vector so that we're sampling as if roughness=1.0

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

    irr_glsl_BSDFSample smpl;
    //==== compute L ====
    H = normalize(m*H);//transform to correct space
    float HdotV = dot(H,interaction.isotropic.V.dir);
    //reflect V on H to actually get L
    smpl.L = H*2.0*HdotV - interaction.isotropic.V.dir;

    //==== compute probability ====
    float a2 = _ax*_ay;
    float lambda = irr_glsl_smith_ggx_Lambda(irr_glsl_smith_ggx_C2(interaction.isotropic.NdotV_squared, a2));
    float G1 = 1.0 / (1.0 + lambda);
    //here using isotropic trowbridge_reitz() instead of irr_glsl_ggx_burley_aniso()
    smpl.probability = irr_glsl_ggx_trowbridge_reitz(a2,NdotH*NdotH) * G1 * abs(dot(interaction.isotropic.V.dir,H)) / interaction.isotropic.NdotV;

    return smpl;
}
irr_glsl_BSDFSample irr_glsl_ggx_smith_cos_gen_sample(in irr_glsl_AnisotropicViewSurfaceInteraction interaction, in uvec2 _sample, in float _ax, in float _ay)
{
    vec2 u = vec2(_sample)/float(UINT_MAX);
    return irr_glsl_ggx_smith_cos_gen_sample(interaction, u, _ax, _ay);
}

#endif
