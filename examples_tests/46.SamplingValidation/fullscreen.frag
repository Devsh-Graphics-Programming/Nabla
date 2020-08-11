#version 430 core

#include <irr/builtin/glsl/bxdf/brdf/specular/ggx.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/beckmann_smith.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>

layout (location = 0) out vec4 Color;

void main()
{
    float ax = 0.5;
    float ay = 0.5;
    vec2 u = gl_FragCoord.xy/vec2(1280.0,720.0);
    irr_glsl_IsotropicViewSurfaceInteraction inter_ = irr_glsl_calcFragmentShaderSurfaceInteraction(vec3(1.0), vec3(0.0), vec3(0.0,0.0,1.0));
    irr_glsl_AnisotropicViewSurfaceInteraction inter = irr_glsl_calcAnisotropicInteraction(inter_, vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    irr_glsl_BSDFSample s = irr_glsl_ggx_cos_generate(inter, u, ax, ay);
    mat2 m = mat2(
        dFdx(s.TdotL),
        dFdx(s.BdotL),

        dFdy(s.TdotL),
        dFdy(s.BdotL)
    );
    float det = determinant(m);

    Color = vec4(1.0/det, 0.0, 0.0, 1.0);
}