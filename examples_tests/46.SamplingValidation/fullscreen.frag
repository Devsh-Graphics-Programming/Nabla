#version 430 core

#extension GL_ARB_derivative_control : enable

#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/beckmann.glsl>
#include <irr/builtin/glsl/bxdf/bsdf/diffuse/lambert.glsl>
#include <irr/builtin/glsl/bxdf/bsdf/specular/ggx.glsl>
//#include <irr/builtin/glsl/bxdf/bsdf/specular/beckmann.glsl>

layout (location = 0) out vec4 Color;

layout (push_constant) uniform PC {
    vec2 a;
    uint test;
} pc;

#define ETC_LAMBERT 0u
#define ETC_GGX 1u
#define ETC_BECKMANN 2u
#define ETC_LAMBERT_TRANSMIT 3u
#define ETC_GGX_TRANSMIT 4u
#define ETC_BECKMANN_TRANSMIT 5u

void main()
{
    const vec2 screenSz = vec2(1280.0,720.0);
    const float ax = pc.a.x;
    const float ay = pc.a.y;
    const mat2x3 ior = mat2x3(vec3(1.02,1.3,1.02), vec3(1.0,2.0,1.0));
    vec3 u = vec3(gl_FragCoord.xy/screenSz,0.0); // random sapling would be useful
    irr_glsl_IsotropicViewSurfaceInteraction inter_ = irr_glsl_calcFragmentShaderSurfaceInteraction(vec3(1.0), vec3(0.0), vec3(0.0,0.0,1.0));
    irr_glsl_AnisotropicViewSurfaceInteraction inter = irr_glsl_calcAnisotropicInteraction(inter_, vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    irr_glsl_LightSample s;
    float pdf = 0.0;
    vec3 rem = vec3(0.0);
    vec3 brdf = vec3(0.0);
    float multiplier = 0.5;
    if (pc.test==ETC_LAMBERT)
    {
        s = irr_glsl_lambertian_cos_generate(inter,u.xy);
        rem = vec3(irr_glsl_lambertian_cos_remainder_and_pdf(pdf, s));
        brdf = vec3(irr_glsl_lambertian_cos_eval(s));
    }
    else if (pc.test==ETC_GGX)
    {
        irr_glsl_AnisotropicMicrofacetCache _cache;
        s = irr_glsl_ggx_cos_generate(inter, u.xy, ax, ay, _cache);
        rem = irr_glsl_ggx_aniso_cos_remainder_and_pdf(pdf, s, inter, _cache, ior, ax, ay);
        brdf = irr_glsl_ggx_height_correlated_aniso_cos_eval(s, inter, _cache, ior, ax, ay);
    }
    else if (pc.test==ETC_BECKMANN)
    {
        irr_glsl_AnisotropicMicrofacetCache _cache;
        s = irr_glsl_beckmann_cos_generate(inter, u.xy, ax, ay, _cache);
        rem = irr_glsl_beckmann_aniso_cos_remainder_and_pdf(pdf, s, inter, _cache, ior, ax, ay);
        brdf = irr_glsl_beckmann_aniso_height_correlated_cos_eval(s, inter, _cache, ior, ax, ay);
    }
    else if (pc.test==ETC_LAMBERT)
    {
        s = irr_glsl_lambertian_transmitter_cos_generate(inter,u);
        rem = vec3(irr_glsl_lambertian_cos_remainder_and_pdf(pdf, s));
        brdf = vec3(irr_glsl_lambertian_cos_eval(s));
        multiplier = 0.25;
    }
    else if (pc.test==ETC_GGX)
    {
        irr_glsl_AnisotropicMicrofacetCache _cache;
        s = irr_glsl_ggx_transmitter_cos_generate(inter, u.xy, ax, ay, _cache);
        rem = irr_glsl_ggx_aniso_cos_remainder_and_pdf(pdf, s, inter, _cache, ior, ax, ay);
        brdf = irr_glsl_ggx_height_correlated_aniso_cos_eval(s, inter, _cache, ior, ax, ay);
        multiplier = 0.25;
    }/*
    else if (pc.test==ETC_BECKMANN)
    {
        irr_glsl_AnisotropicMicrofacetCache _cache;
        s = irr_glsl_beckmann_transmitter_cos_generate(inter, u.xy, ax, ay, _cache);
        rem = irr_glsl_beckmann_aniso_cos_remainder_and_pdf(pdf, s, inter, _cache, ior, ax, ay);
        brdf = irr_glsl_beckmann_aniso_height_correlated_cos_eval(s, inter, _cache, ior, ax, ay);
        multiplier = 0.25;
    }*/

    mat2 m = mat2(
        dFdxFine(s.TdotL)*screenSz.x,
        dFdxFine(s.BdotL)*screenSz.x,

        dFdyFine(s.TdotL)*screenSz.y,
        dFdyFine(s.BdotL)*screenSz.y
    );
    float det = determinant(m);

    Color = vec4(abs(rem*pdf-brdf),multiplier*abs(det*pdf/s.NdotL)); // preferred version of the test
    //Color = vec4(multiplier*abs(det*pdf/s.NdotL),abs(rem*pdf-brdf));
}