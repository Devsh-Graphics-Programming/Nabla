// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_FRAG_INPUTS_DEFINED_
#define _NBL_FRAG_INPUTS_DEFINED_
layout (location = 0) in vec3 LocalPos;
layout (location = 1) in vec3 ViewPos;
layout (location = 2) in vec3 Normal;
#ifndef _NO_UV
layout (location = 3) in vec2 UV;
#endif
#endif //_NBL_FRAG_INPUTS_DEFINED_

#ifndef _NBL_FRAG_OUTPUTS_DEFINED_
#define _NBL_FRAG_OUTPUTS_DEFINED_
layout (location = 0) out vec4 OutColor;
#endif //_NBL_FRAG_OUTPUTS_DEFINED_

#define ILLUM_MODEL_MASK 0x0fu
#define map_Ka_MASK uint(1u<<4u)
#define map_Kd_MASK uint(1u<<5u)
#define map_Ks_MASK uint(1u<<6u)
#define map_Ns_MASK uint(1u<<8u)
#define map_d_MASK uint(1u<<9u)
#define map_bump_MASK uint(1u<<10u)
#define map_normal_MASK uint(1u<<11u)

#include <nbl/builtin/glsl/bxdf/common.glsl> // change to bxdf/common.glsl

#include <nbl/builtin/glsl/loader/mtl/common.glsl>

#ifndef _NBL_FRAG_PUSH_CONSTANTS_DEFINED_
#define _NBL_FRAG_PUSH_CONSTANTS_DEFINED_

layout (push_constant) uniform Block {
    nbl_glsl_MTLMaterialParameters params;
} PC;
#endif //_NBL_FRAG_PUSH_CONSTANTS_DEFINED_

#ifndef _NBL_FRAG_GET_MATERIAL_PARAMETERS_FUNCTION_DEFINED_
#define _NBL_FRAG_GET_MATERIAL_PARAMETERS_FUNCTION_DEFINED_

nbl_glsl_MTLMaterialParameters nbl_glsl_getMaterialParameters()
{
    return PC.params;
}
#endif //_NBL_FRAG_RETRIVE_MATERIAL_PARAMETERS_FUNCTION_DEFINED_

#if !defined(_NBL_FRAG_SET3_BINDINGS_DEFINED_) && !defined(_NO_UV)
#define _NBL_FRAG_SET3_BINDINGS_DEFINED_
layout (set = 3, binding = 0) uniform sampler2D map_Ka;
layout (set = 3, binding = 1) uniform sampler2D map_Kd;
layout (set = 3, binding = 2) uniform sampler2D map_Ks;
layout (set = 3, binding = 4) uniform sampler2D map_Ns;
layout (set = 3, binding = 5) uniform sampler2D map_d;
layout (set = 3, binding = 6) uniform sampler2D map_bump;
#endif //_NBL_FRAG_SET3_BINDINGS_DEFINED_

#if !defined(_NBL_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_) && !defined(_NO_UV)
#ifndef _NBL_Ka_SAMPLE_FUNCTION_DEFINED_
vec4 nbl_sample_Ka(in vec2 uv, in mat2 dUV) { return texture(map_Ka, uv); }
#endif
#ifndef _NBL_Kd_SAMPLE_FUNCTION_DEFINED_
vec4 nbl_sample_Kd(in vec2 uv, in mat2 dUV) { return texture(map_Kd, uv); }
#endif
#ifndef _NBL_Ks_SAMPLE_FUNCTION_DEFINED_
vec4 nbl_sample_Ks(in vec2 uv, in mat2 dUV) { return texture(map_Ks, uv); }
#endif
#ifndef _NBL_Ns_SAMPLE_FUNCTION_DEFINED_
vec4 nbl_sample_Ns(in vec2 uv, in mat2 dUV) { return texture(map_Ns, uv); }
#endif
#ifndef _NBL_d_SAMPLE_FUNCTION_DEFINED_
vec4 nbl_sample_d(in vec2 uv, in mat2 dUV) { return texture(map_d, uv); }
#endif
#ifndef _NBL_bump_SAMPLE_FUNCTION_DEFINED_
vec4 nbl_sample_bump(in vec2 uv, in mat2 dUV) { return texture(map_bump, uv) * 2.f - vec4(1.f); }
#endif
#endif //_NBL_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_


#include <nbl/builtin/glsl/bxdf/fresnel.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/diffuse/fresnel_correction.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/diffuse/lambert.glsl>
#include <nbl/builtin/glsl/bxdf/brdf/specular/blinn_phong.glsl>

#ifndef _NBL_BSDF_COS_EVAL_DEFINED_
#define _NBL_BSDF_COS_EVAL_DEFINED_

// Spectrum can be exchanged to a float for monochrome
#define Spectrum vec3

//! This is the function that evaluates the BSDF for specific view and observer direction
// params can be either BSDFIsotropicParams or BSDFAnisotropicParams 
Spectrum nbl_bsdf_cos_eval(in nbl_glsl_LightSample _sample, in nbl_glsl_IsotropicViewSurfaceInteraction inter, in mat2 dUV)
{
    nbl_glsl_MTLMaterialParameters mtParams = nbl_glsl_getMaterialParameters();

    vec3 Kd;
#ifndef _NO_UV
    if ((mtParams.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Kd = nbl_sample_Kd(UV, dUV).rgb;
    else
#endif
        Kd = mtParams.Kd;

    vec3 color = vec3(0.0);
    vec3 Ks;
    float Ns;

#ifndef _NO_UV
    if ((mtParams.extra&(map_Ks_MASK)) == (map_Ks_MASK))
        Ks = nbl_sample_Ks(UV, dUV).rgb;
    else
#endif
        Ks = mtParams.Ks;
#ifndef _NO_UV
    if ((mtParams.extra&(map_Ns_MASK)) == (map_Ns_MASK))
        Ns = nbl_sample_Ns(UV, dUV).x;
    else
#endif
        Ns = mtParams.Ns;

    vec3 Ni = vec3(mtParams.Ni);


    vec3 diff = nbl_glsl_lambertian_cos_eval(_sample) * Kd * (1.0-nbl_glsl_fresnel_dielectric(Ni,_sample.NdotL)) * (1.0-nbl_glsl_fresnel_dielectric(Ni,inter.NdotV));
    diff *= nbl_glsl_diffuseFresnelCorrectionFactor(Ni, Ni*Ni);

    nbl_glsl_IsotropicMicrofacetCache _cache = nbl_glsl_calcIsotropicMicrofacetCache(inter,_sample);
    switch (mtParams.extra&ILLUM_MODEL_MASK)
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
        vec3 spec = Ks*nbl_glsl_blinn_phong_cos_eval(_sample, inter, _cache, Ns, mat2x3(Ni,vec3(0.0)));
        color = (diff + spec);
    }
        break;
    case 4:
    case 6:
    case 7:
    case 9://basically same as 4
    {
        vec3 spec = Ks*nbl_glsl_blinn_phong_cos_eval(_sample, inter, _cache, Ns, mat2x3(Ni,vec3(0.0)));
        color = spec;
    }
        break;
    default:
        break;
    }

    return color;  
}

#endif //_NBL_BSDF_COS_EVAL_DEFINED_

#include <nbl/builtin/glsl/bump_mapping/utils.glsl>

#ifndef _NBL_COMPUTE_LIGHTING_DEFINED_
#define _NBL_COMPUTE_LIGHTING_DEFINED_

vec3 nbl_computeLighting(out nbl_glsl_IsotropicViewSurfaceInteraction out_interaction, in mat2 dUV)
{
    nbl_glsl_IsotropicViewSurfaceInteraction interaction = nbl_glsl_calcSurfaceInteraction(vec3(0.0), ViewPos, Normal, mat2x3(dFdx(ViewPos),dFdy(ViewPos)));
    nbl_glsl_MTLMaterialParameters mtParams = nbl_glsl_getMaterialParameters();

#ifndef _NO_UV
    if ((mtParams.extra&map_bump_MASK) == map_bump_MASK)
    {
        interaction.N = normalize(interaction.N);

        vec2 dh = nbl_sample_bump(UV, dUV).xy;

        interaction.N = nbl_glsl_perturbNormal_derivativeMap(interaction.N, dh, interaction.V.dPosdScreen, dUV);
    }
#endif
    const vec3 L = -ViewPos;
    const float lenL2 = dot(L,L);
    const float invLenL = inversesqrt(lenL2);
    nbl_glsl_LightSample _sample = nbl_glsl_createLightSample(L*invLenL, interaction);

    vec3 Ka;
    switch ((mtParams.extra&ILLUM_MODEL_MASK))
    {
    case 0:
    {
#ifndef _NO_UV
    if ((mtParams.extra&(map_Kd_MASK)) == (map_Kd_MASK))
        Ka = nbl_sample_bump(UV, dUV).rgb;
    else
#endif
        Ka = mtParams.Kd;
    }
    break;
    default:
#define Ia 0.1
    {
#ifndef _NO_UV
    if ((mtParams.extra&(map_Ka_MASK)) == (map_Ka_MASK))
        Ka = nbl_sample_Ka(UV, dUV).rgb;
    else
#endif
        Ka = mtParams.Ka;
    Ka *= Ia;
    }
#undef Ia
    break;
    }

    out_interaction = interaction;
#define Intensity 1000000.0
    return (Intensity/lenL2)*nbl_bsdf_cos_eval(_sample,interaction, dUV) + Ka;
#undef Intensity
}
#endif //_NBL_COMPUTE_LIGHTING_DEFINED_


#ifndef _NBL_FRAG_MAIN_DEFINED_
#define _NBL_FRAG_MAIN_DEFINED_

void main()
{
#ifndef _NO_UV
    mat2 dUV = mat2(dFdx(UV), dFdy(UV));    
#else
    mat2 dUV = mat2(vec2(0,0),vec2(0,0));    
#endif
    nbl_glsl_MTLMaterialParameters mtParams = nbl_glsl_getMaterialParameters();
    nbl_glsl_IsotropicViewSurfaceInteraction interaction;
    vec3 color = nbl_computeLighting(interaction, dUV);

    float d = mtParams.d;

    //another illum model switch, required for illum=4,6,7,9 to compute alpha from fresnel (taken from opacity map or constant otherwise)
    switch (mtParams.extra&ILLUM_MODEL_MASK)
    {
        case 4:
        case 6:
        case 7:
        case 9:
        {
            float VdotN = dot(interaction.N, interaction.V.dir);
            d = nbl_glsl_fresnel_dielectric(vec3(mtParams.Ni), VdotN).x;
        }
            break;
        default:
    #ifndef _NO_UV
            if ((mtParams.extra&(map_d_MASK)) == (map_d_MASK))
            {
                d = nbl_sample_d(UV, dUV).r;
                color *= d;
            }
    #endif
            break;
    }

    OutColor = vec4(color, d);
}
#endif //_NBL_FRAG_MAIN_DEFINED_