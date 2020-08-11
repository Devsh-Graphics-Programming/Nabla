#version 430 core

layout (location = 0) in vec3 Normal;
layout (location = 1) in vec3 Pos;
layout (location = 2) flat in float Alpha;

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PC {
    layout (offset = 64) vec3 campos;
} pc;

#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/beckmann_smith.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>

void main()
{
    const vec3 lightPos = vec3(6.75, 4.0, -1.0);
    const float Intensity = 20.0;

    vec3 L = lightPos-Pos;
    vec3 Lnorm = normalize(L);
    //vec3 L = normalize(vec3(1.0,3.0,1.0));
    const float a2 = Alpha*Alpha;
    const float ax = Alpha;
    const float ay = Alpha;
    vec3 N = normalize(Normal);
    if (dot(N,Lnorm)>0.0)
    {
        irr_glsl_IsotropicViewSurfaceInteraction inter_ = irr_glsl_calcFragmentShaderSurfaceInteraction(pc.campos, Pos, N);
        irr_glsl_AnisotropicViewSurfaceInteraction inter = irr_glsl_calcAnisotropicInteraction(inter_);
        irr_glsl_BSDFIsotropicParams params_ = irr_glsl_calcBSDFIsotropicParams(inter_, Lnorm);
        irr_glsl_BSDFAnisotropicParams params = irr_glsl_calcBSDFAnisotropicParams(params_, inter);
        const mat2x3 ior = mat2x3(vec3(1.02,1.3,1.02), vec3(1.0,2.0,1.0));
        const vec3 albedo = vec3(0.5);

        vec3 brdf = vec3(0.0);

//when TEST_GGX_SMITH is defined: 
//key 1 = iso ggx smith
//key 2 = aniso ggx smith
//#define TEST_GGX_SMITH

#ifdef TEST_GGX
    #ifdef TEST_GGX_SMITH
        brdf = vec3( irr_glsl_ggx_smith_height_correlated(a2, params_.NdotL, inter_.NdotV) );
        //brdf *= irr_glsl_fresnel_conductor(ior[0], ior[1], params.isotropic.VdotH);
    #else
        brdf = irr_glsl_ggx_height_correlated_cos_eval(params_, inter_, ior, a2);
    #endif
#elif defined(TEST_BECKMANN)
        //brdf = irr_glsl_ggx_height_correlated_aniso_cos_eval2(params, inter, ior, Alpha, Alpha);
    #ifdef TEST_GGX_SMITH
        brdf = vec3( 4.0*params_.NdotL*inter_.NdotV*irr_glsl_ggx_smith_height_correlated_aniso_wo_numerator(ax, ay, params.TdotL, inter.TdotV, params.BdotL, inter.BdotV, params.isotropic.NdotL, inter.isotropic.NdotV) );
        //brdf *= irr_glsl_fresnel_conductor(ior[0], ior[1], params.isotropic.VdotH);
    #else
        brdf = irr_glsl_beckmann_smith_height_correlated_cos_eval(params_, inter_, ior, a2);
    #endif
#elif defined(TEST_PHONG)
        float n = 2.0/a2 - 2.0;//conversion between alpha and Phong exponent, Walter et.al.
        brdf = irr_glsl_blinn_phong_fresnel_conductor_cos_eval(params_, inter_, n, ior);
#elif defined(TEST_AS)
    #error "Not implemented"
#elif defined(TEST_OREN_NAYAR)
    brdf = albedo*irr_glsl_oren_nayar_cos_eval(params_, inter_, a2);
#elif defined(TEST_LAMBERT)
    brdf = albedo*irr_glsl_lambertian_cos_eval(params_, inter_);
#endif
    //red output means brdf>1.0
    outColor = any(greaterThan(brdf,vec3(1.0))) ? vec4(1.0,0.0,0.0,1.0) : vec4(Intensity*brdf/dot(L,L), 1.0);
    //outColor = vec4(Intensity*brdf/*/dot(L,L)*/, 1.0);
    //outColor = (inter_.NdotV<0.0||params_.NdotL<0.0) ? vec4(1.0,0.0,0.0,1.0) : vec4(Intensity*brdf/*/dot(L,L)*/, 1.0);
    }
    else outColor = vec4(0.0,0.0,0.0,1.0);
}