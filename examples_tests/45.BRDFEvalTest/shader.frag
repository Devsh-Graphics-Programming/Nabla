#version 430 core
// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


layout (location = 0) in vec3 Normal;
layout (location = 1) in vec3 Pos;
layout (location = 2) flat in float Alpha;

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PC {
    layout (offset = 64) vec3 campos;
} pc;

#include <nbl/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/beckmann.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/diffuse/oren_nayar.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>

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
        nbl_glsl_IsotropicViewSurfaceInteraction inter_ = nbl_glsl_calcSurfaceInteraction(pc.campos, Pos, N);
        nbl_glsl_AnisotropicViewSurfaceInteraction inter = nbl_glsl_calcAnisotropicInteraction(inter_);

        nbl_glsl_LightSample _sample = nbl_glsl_createLightSample(Lnorm,inter);

        nbl_glsl_AnisotropicMicrofacetCache cache = nbl_glsl_calcAnisotropicMicrofacetCache(inter, _sample);

        const mat2x3 ior = mat2x3(vec3(1.02,1.3,1.02), vec3(1.0,2.0,1.0));
        const vec3 albedo = vec3(0.5);

        vec3 brdf = vec3(0.0);

#ifdef TEST_GGX
        brdf = nbl_glsl_ggx_height_correlated_cos_eval(_sample, inter_, cache.isotropic, ior, a2);
#elif defined(TEST_BECKMANN)
        brdf = nbl_glsl_beckmann_height_correlated_cos_eval(_sample, inter_, cache.isotropic, ior, a2);
#elif defined(TEST_PHONG)
        float n = nbl_glsl_alpha2_to_phong_exp(a2);
        brdf = nbl_glsl_blinn_phong_cos_eval(_sample, inter_, cache.isotropic, n, ior);
#elif defined(TEST_AS)
        float nx = nbl_glsl_alpha2_to_phong_exp(a2);
        float aa = 1.0-Alpha;
        float ny = nbl_glsl_alpha2_to_phong_exp(aa*aa);
        brdf = nbl_glsl_blinn_phong_cos_eval(_sample, inter, cache, nx, ny, ior);
#elif defined(TEST_OREN_NAYAR)
        brdf = albedo*nbl_glsl_oren_nayar_cos_eval(_sample, inter_, a2);
#elif defined(TEST_LAMBERT)
        brdf = albedo*nbl_glsl_lambertian_cos_eval(_sample);
#endif
        const vec3 col = Intensity*brdf/dot(L,L);
        //red output means brdf>1.0
        //outColor = any(greaterThan(brdf,vec3(1.0))) ? vec4(1.0,0.0,0.0,1.0) : vec4(Intensity*brdf/dot(L,L), 1.0);
        outColor = vec4(col, 1.0);
        //outColor = (inter_.NdotV<0.0||_sample.NdotL<0.0) ? vec4(1.0,0.0,0.0,1.0) : vec4(col, 1.0);
    }
    else outColor = vec4(0.0,0.0,0.0,1.0);
}