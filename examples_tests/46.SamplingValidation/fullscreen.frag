// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core

#extension GL_ARB_derivative_control : enable

#include <nbl/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/beckmann.glsl>
#include <nbl/builtin/glsl/bxdf/bsdf/diffuse/lambert.glsl>
//#include <nbl/builtin/glsl/bxdf/bsdf/specular/ggx.glsl>
//#include <nbl/builtin/glsl/bxdf/bsdf/specular/beckmann.glsl>

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
    nbl_glsl_IsotropicViewSurfaceInteraction inter_ = nbl_glsl_calcSurfaceInteraction(vec3(1.0), vec3(0.0), vec3(0.0,0.0,1.0));
    nbl_glsl_AnisotropicViewSurfaceInteraction inter = nbl_glsl_calcAnisotropicInteraction(inter_, vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    nbl_glsl_LightSample s;
    float pdf = 0.0;
    vec3 rem = vec3(0.0);
    vec3 brdf = vec3(0.0);
    if (pc.test==ETC_LAMBERT)
    {
        s = nbl_glsl_lambertian_cos_generate(inter,u.xy);
        rem = vec3(nbl_glsl_lambertian_cos_remainder_and_pdf(pdf, s));
        brdf = vec3(nbl_glsl_lambertian_cos_eval(s));
    }
    else if (pc.test==ETC_GGX)
    {
        nbl_glsl_AnisotropicMicrofacetCache _cache;
        s = nbl_glsl_ggx_cos_generate(inter, u.xy, ax, ay, _cache);
        rem = nbl_glsl_ggx_aniso_cos_remainder_and_pdf(pdf, s, inter, _cache, ior, ax, ay);
        brdf = nbl_glsl_ggx_height_correlated_aniso_cos_eval(s, inter, _cache, ior, ax, ay);
    }
    else if (pc.test==ETC_BECKMANN)
    {
        nbl_glsl_AnisotropicMicrofacetCache _cache;
        s = nbl_glsl_beckmann_cos_generate(inter, u.xy, ax, ay, _cache);
        rem = nbl_glsl_beckmann_aniso_cos_remainder_and_pdf(pdf, s, inter, _cache, ior, ax, ay);
        brdf = nbl_glsl_beckmann_aniso_height_correlated_cos_eval(s, inter, _cache, ior, ax, ay);
    /*
        s = nbl_glsl_beckmann_cos_generate(inter,u.xy,ax,ay,_cache);
        rem = vec3(1.0);//4.0/inter.isotropic.NdotV);
        pdf = nbl_glsl_beckmann_pdf_wo_clamps(_cache.isotropic.NdotH2,max(inter.isotropic.NdotV,0.0),inter.isotropic.NdotV_squared,ax*ay);
        brdf = vec3(pdf);
        */
    }
    else if (pc.test==ETC_LAMBERT)
    {
        s = nbl_glsl_lambertian_transmitter_cos_generate(inter,u);
        rem = vec3(nbl_glsl_lambertian_cos_remainder_and_pdf(pdf, s));
        brdf = vec3(nbl_glsl_lambertian_cos_eval(s));
    }/* cant do the tests properly :(, would need a 3D derivative and determinant
    else if (pc.test==ETC_GGX)
    {
        nbl_glsl_AnisotropicMicrofacetCache _cache;
        s = nbl_glsl_ggx_transmitter_cos_generate(inter, u.xy, ax, ay, _cache);
        rem = nbl_glsl_ggx_transmitter_aniso_cos_remainder_and_pdf(pdf, s, inter, _cache, ior[0].g, ax, ay);
        brdf = nbl_glsl_ggx_transmitter_height_correlated_aniso_cos_eval(s, inter, _cache, ior[0].g, ax, ay);
        multiplier = 0.25;
    }
    else if (pc.test==ETC_BECKMANN)
    {
        nbl_glsl_AnisotropicMicrofacetCache _cache;
        s = nbl_glsl_beckmann_transmitter_cos_generate(inter, u.xy, ax, ay, _cache);
        rem = nbl_glsl_beckmann_aniso_cos_remainder_and_pdf(pdf, s, inter, _cache, ior[0].g, ax, ay);
        brdf = nbl_glsl_beckmann_aniso_height_correlated_cos_eval(s, inter, _cache, ior[0].g, ax, ay);
        multiplier = 0.25;
    }*/

    mat2 m = mat2(
        dFdxFine(s.TdotL)*screenSz.x,
        dFdxFine(s.BdotL)*screenSz.x,

        dFdyFine(s.TdotL)*screenSz.y,
        dFdyFine(s.BdotL)*screenSz.y
    );
    float det = determinant(m);

    const vec4 validColor = vec4(0.0,0.0,0.0,0.5); 
    if (s.NdotL>0.0)
    {
        Color = vec4(abs(rem*pdf-brdf),abs(det*pdf/s.NdotL)*0.5); // preferred version of the test
        //Color = vec4(abs(det*pdf/s.NdotL)*0.5,abs(rem*pdf-brdf));
    }
    else
        Color = validColor;
}