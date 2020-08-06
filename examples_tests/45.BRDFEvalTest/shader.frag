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

void main()
{
    const vec3 lightPos = vec3(6.75, 4.0, -1.0);
    const float Intensity = 20.0;

    vec3 L = normalize(lightPos-Pos);
    float a2 = Alpha*Alpha;
    vec3 N = normalize(Normal);
    if (dot(N,L)>0.0)
    {
        irr_glsl_IsotropicViewSurfaceInteraction inter = irr_glsl_calcFragmentShaderSurfaceInteraction(pc.campos, Pos, N);
        irr_glsl_BSDFIsotropicParams params = irr_glsl_calcBSDFIsotropicParams(inter, L);
        mat2x3 ior = mat2x3(vec3(1.02,1.3,1.02), vec3(1.0,2.0,1.0));
        vec3 brdf;
#ifdef TEST_GGX
        brdf = irr_glsl_ggx_height_correlated_cos_eval(params, inter, ior, a2);
#elif defined(TEST_BECKMANN)
        brdf = irr_glsl_beckmann_smith_height_correlated_cos_eval(params, inter, ior, a2);
#elif defined(TEST_PHONG)
        float n = max(2.0/a2 - 2.0, 0.0);
        brdf = irr_glsl_blinn_phong_fresnel_conductor_cos_eval(params, inter, n, ior);
#elif defined(TEST_AS)
    #error "Not implemented"
#endif
    outColor = vec4(Intensity*brdf/dot(L,L), 1.0);
    }
    else outColor = vec4(0.0,0.0,0.0,1.0);
}