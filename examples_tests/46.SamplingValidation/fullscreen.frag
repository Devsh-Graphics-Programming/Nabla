#version 430 core

#extension GL_ARB_derivative_control : enable

#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/beckmann_smith.glsl>
#include <irr/builtin/glsl/bxdf/brdf/cos_weighted_sample.glsl>

layout (location = 0) out vec4 Color;

layout (push_constant) uniform PC {
    vec2 a;
    uint test;
} pc;
#define ETC_COS_WEIGHTED 0u
#define ETC_GGX 1u
#define ETC_BECKMANN 2u

void main()
{
    const vec2 screenSz = vec2(1280.0,720.0);
    const float ax = pc.a.x;
    const float ay = pc.a.y;
    const mat2x3 ior = mat2x3(vec3(1.02,1.3,1.02), vec3(1.0,2.0,1.0));
    vec2 u = gl_FragCoord.xy/screenSz;
    irr_glsl_IsotropicViewSurfaceInteraction inter_ = irr_glsl_calcFragmentShaderSurfaceInteraction(vec3(1.0), vec3(0.0), vec3(0.0,0.0,1.0));
    irr_glsl_AnisotropicViewSurfaceInteraction inter = irr_glsl_calcAnisotropicInteraction(inter_, vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    irr_glsl_BSDFIsotropicParams params_;
    irr_glsl_BSDFAnisotropicParams params;
    irr_glsl_BSDFSample s;
    float pdf = 0.0;
    vec3 rem = vec3(0.0);
    vec3 brdf = vec3(0.0);
    if (pc.test==ETC_COS_WEIGHTED)
    {
        s = irr_glsl_cos_weighted_cos_generate(inter, u);
        rem = irr_glsl_cos_weighted_cos_remainder_and_pdf(pdf, s, inter_);
        brdf = vec3(s.NdotL*irr_glsl_RECIPROCAL_PI);
    }
    else if (pc.test==ETC_GGX)
    {
        params_ = irr_glsl_calcBSDFIsotropicParams(inter_, s.L);
        params = irr_glsl_calcBSDFAnisotropicParams(params_, inter);

        s = irr_glsl_ggx_cos_generate(inter, u, ax, ay);
        rem = irr_glsl_ggx_aniso_cos_remainder_and_pdf(pdf, s, inter, ior, ax, ay);
        brdf = irr_glsl_ggx_height_correlated_aniso_cos_eval(params, inter, ior, ax, ay);
    }
    else if (pc.test==ETC_BECKMANN)
    {
        params_ = irr_glsl_calcBSDFIsotropicParams(inter_, s.L);
        params = irr_glsl_calcBSDFAnisotropicParams(params_, inter);

        s = irr_glsl_beckmann_smith_cos_generate(inter, u, ax, ay);
        rem = irr_glsl_beckmann_aniso_cos_remainder_and_pdf(pdf, s, inter, ior, ax, ax*ax, ay, ay*ay);
        brdf = irr_glsl_beckmann_aniso_smith_height_correlated_cos_eval(params, inter, ior, ax, ax*ax, ay, ay*ay);
    }

    mat2 m = mat2(
        dFdxFine(s.TdotL)*screenSz.x,
        dFdxFine(s.BdotL)*screenSz.x,

        dFdyFine(s.TdotL)*screenSz.y,
        dFdyFine(s.BdotL)*screenSz.y
    );
    float det = determinant(m);

    //Color = vec4(abs(rem*pdf-brdf),0.5*abs(det*pdf/s.NdotL)); // preferred version of the test
    Color = vec4(0.5*abs(det*pdf/s.NdotL),abs(rem*pdf-brdf));
}