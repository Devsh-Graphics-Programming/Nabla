// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _IRR_FRAG_INPUTS_DEFINED_
#define _IRR_FRAG_INPUTS_DEFINED_
layout (location = 0) in vec3 LocalPos;
layout (location = 1) in vec3 ViewPos;
layout (location = 2) in vec3 Normal;
#ifndef _NO_UV
layout (location = 3) in vec2 UV;
#endif
#endif //_IRR_FRAG_INPUTS_DEFINED_

#ifndef _IRR_FRAG_OUTPUTS_DEFINED_
#define _IRR_FRAG_OUTPUTS_DEFINED_
layout (location = 0) out vec4 OutColor;
#endif //_IRR_FRAG_OUTPUTS_DEFINED_

#define ILLUM_MODEL_MASK 0x0fu
#define map_Ka_MASK uint(1u<<4u)
#define map_Kd_MASK uint(1u<<5u)
#define map_Ks_MASK uint(1u<<6u)
#define map_Ns_MASK uint(1u<<8u)
#define map_d_MASK uint(1u<<9u)
#define map_bump_MASK uint(1u<<10u)
#define map_normal_MASK uint(1u<<11u)

#include <irr/builtin/glsl/bxdf/common.glsl> // change to bxdf/common.glsl

#ifndef _IRR_FRAG_PUSH_CONSTANTS_DEFINED_
#define _IRR_FRAG_PUSH_CONSTANTS_DEFINED_

#include <irr/builtin/glsl/loaders/mtl/common.glsl>

layout (push_constant) uniform Block {
    irr_glsl_MTLMaterialParameters params;
} PC;
#endif //_IRR_FRAG_PUSH_CONSTANTS_DEFINED_

#if !defined(_IRR_FRAG_SET3_BINDINGS_DEFINED_) && !defined(_NO_UV)
#define _IRR_FRAG_SET3_BINDINGS_DEFINED_
layout (set = 3, binding = 0) uniform sampler2D map_Ka;
layout (set = 3, binding = 1) uniform sampler2D map_Kd;
layout (set = 3, binding = 2) uniform sampler2D map_Ks;
layout (set = 3, binding = 4) uniform sampler2D map_Ns;
layout (set = 3, binding = 5) uniform sampler2D map_d;
layout (set = 3, binding = 6) uniform sampler2D map_bump;
#endif //_IRR_FRAG_SET3_BINDINGS_DEFINED_

#if !defined(_IRR_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_) && !defined(_NO_UV)
#ifndef _IRR_Ka_SAMPLE_FUNCTION_DEFINED_
vec4 irr_sample_Ka(in vec2 uv, in mat2 dUV) { return texture(map_Ka, uv); }
#endif
#ifndef _IRR_Kd_SAMPLE_FUNCTION_DEFINED_
vec4 irr_sample_Kd(in vec2 uv, in mat2 dUV) { return texture(map_Kd, uv); }
#endif
#ifndef _IRR_Ks_SAMPLE_FUNCTION_DEFINED_
vec4 irr_sample_Ks(in vec2 uv, in mat2 dUV) { return texture(map_Ks, uv); }
#endif
#ifndef _IRR_Ns_SAMPLE_FUNCTION_DEFINED_
vec4 irr_sample_Ns(in vec2 uv, in mat2 dUV) { return texture(map_Ns, uv); }
#endif
#ifndef _IRR_d_SAMPLE_FUNCTION_DEFINED_
vec4 irr_sample_d(in vec2 uv, in mat2 dUV) { return texture(map_d, uv); }
#endif
#ifndef _IRR_bump_SAMPLE_FUNCTION_DEFINED_
vec4 irr_sample_bump(in vec2 uv, in mat2 dUV) { return texture(map_bump, uv); }
#endif
#endif //_IRR_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_


#include <irr/builtin/glsl/bxdf/fresnel.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl>
#include <irr/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
#include <irr/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>

#ifndef _IRR_BSDF_COS_EVAL_DEFINED_
#define _IRR_BSDF_COS_EVAL_DEFINED_

// Spectrum can be exchanged to a float for monochrome
#define Spectrum vec3

//! This is the function that evaluates the BSDF for specific view and observer direction
// params can be either BSDFIsotropicParams or BSDFAnisotropicParams 
Spectrum irr_bsdf_cos_eval(in irr_glsl_LightSample _sample, in irr_glsl_IsotropicViewSurfaceInteraction inter, in mat2 dUV)
{
    vec3 Kd;
#ifndef _NO_UV
    if ((PC.params.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Kd = irr_sample_Kd(UV, dUV).rgb;
    else
#endif
        Kd = PC.params.Kd;

    vec3 color = vec3(0.0);
    vec3 Ks;
    float Ns;

#ifndef _NO_UV
    if ((PC.params.extra&(map_Ks_MASK)) == (map_Ks_MASK))
        Ks = irr_sample_Ks(UV, dUV).rgb;
    else
#endif
        Ks = PC.params.Ks;
#ifndef _NO_UV
    if ((PC.params.extra&(map_Ns_MASK)) == (map_Ns_MASK))
        Ns = irr_sample_Ns(UV, dUV).x;
    else
#endif
        Ns = PC.params.Ns;

    vec3 Ni = vec3(PC.params.Ni);


    vec3 diff = irr_glsl_lambertian_cos_eval(_sample) * Kd * (1.0-irr_glsl_fresnel_dielectric(Ni,_sample.NdotL)) * (1.0-irr_glsl_fresnel_dielectric(Ni,inter.NdotV));
    diff *= irr_glsl_diffuseFresnelCorrectionFactor(Ni, Ni*Ni);

    irr_glsl_IsotropicMicrofacetCache _cache = irr_glsl_calcIsotropicMicrofacetCache(inter,_sample);
    switch (PC.params.extra&ILLUM_MODEL_MASK)
    {
    case 0:
        color = vec3(0.0);
        break;
    case 1:
        color = diff;
        break;
    case 2:
    case 3://2 + IBL
    case 5://basically same as 3
    case 8://basically same as 5
    {
        vec3 spec = Ks*irr_glsl_blinn_phong_cos_eval(_sample, inter, _cache, Ns, mat2x3(Ni,vec3(0.0)));
        color = (diff + spec);
    }
        break;
    case 4:
    case 6:
    case 7:
    case 9://basically same as 4
    {
        vec3 spec = Ks*irr_glsl_blinn_phong_cos_eval(_sample, inter, _cache, Ns, mat2x3(Ni,vec3(0.0)));
        color = spec;
    }
        break;
    default:
        break;
    }

    return color;  
}

#endif //_IRR_BSDF_COS_EVAL_DEFINED_

#include <irr/builtin/glsl/bump_mapping/utils.glsl>

#ifndef _IRR_COMPUTE_LIGHTING_DEFINED_
#define _IRR_COMPUTE_LIGHTING_DEFINED_

vec3 irr_computeLighting(out irr_glsl_IsotropicViewSurfaceInteraction out_interaction, in mat2 dUV)
{
    irr_glsl_IsotropicViewSurfaceInteraction interaction = irr_glsl_calcFragmentShaderSurfaceInteraction(vec3(0.0), ViewPos, Normal);

#ifndef _NO_UV
    if ((PC.params.extra&map_bump_MASK) == map_bump_MASK)
    {
        interaction.N = normalize(interaction.N);

        float h = irr_sample_bump(UV, dUV).x;

        vec2 dHdScreen = vec2(dFdx(h), dFdy(h));
        interaction.N = irr_glsl_perturbNormal_heightMap(interaction.N, interaction.V.dPosdScreen, dHdScreen);
    }
#endif
    const vec3 L = -ViewPos;
    const float lenL2 = dot(L,L);
    const float invLenL = inversesqrt(lenL2);
    irr_glsl_LightSample _sample = irr_glsl_createLightSample(L*invLenL, interaction);

    vec3 Ka;
    switch ((PC.params.extra&ILLUM_MODEL_MASK))
    {
    case 0:
    {
#ifndef _NO_UV
    if ((PC.params.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Ka = irr_sample_bump(UV, dUV).rgb;
    else
#endif
        Ka = PC.params.Kd;
    }
    break;
    default:
#define Ia 0.1
    {
#ifndef _NO_UV
    if ((PC.params.extra&(map_Ka_MASK)) == (map_Ka_MASK))
        Ka = irr_sample_Ka(UV, dUV).rgb;
    else
#endif
        Ka = PC.params.Ka;
    Ka *= Ia;
    }
#undef Ia
    break;
    }

    out_interaction = interaction;
#define Intensity 1000000.0
    return (Intensity/lenL2)*irr_bsdf_cos_eval(_sample,interaction, dUV) + Ka;
#undef Intensity
}
#endif //_IRR_COMPUTE_LIGHTING_DEFINED_


#ifndef _IRR_FRAG_MAIN_DEFINED_
#define _IRR_FRAG_MAIN_DEFINED_

void main()
{
#ifndef _NO_UV
    mat2 dUV = mat2(dFdx(UV), dFdy(UV));    
#else
    mat2 dUV = mat2(vec2(0,0),vec2(0,0));    
#endif
    irr_glsl_IsotropicViewSurfaceInteraction interaction;
    vec3 color = irr_computeLighting(interaction, dUV);

    float d = PC.params.d;

    //another illum model switch, required for illum=4,6,7,9 to compute alpha from fresnel (taken from opacity map or constant otherwise)
    switch (PC.params.extra&ILLUM_MODEL_MASK)
    {
        case 4:
        case 6:
        case 7:
        case 9:
        {
            float VdotN = dot(interaction.N, interaction.V.dir);
            d = irr_glsl_fresnel_dielectric(vec3(PC.params.Ni), VdotN).x;
        }
            break;
        default:
    #ifndef _NO_UV
            if ((PC.params.extra&(map_d_MASK)) == (map_d_MASK))
            {
                d = irr_sample_d(UV, dUV).r;
                color *= d;
            }
    #endif
            break;
    }

    OutColor = vec4(color, d);
}
#endif //_IRR_FRAG_MAIN_DEFINED_