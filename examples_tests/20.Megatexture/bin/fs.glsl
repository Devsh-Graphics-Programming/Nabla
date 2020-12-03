// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#version 430 core
#ifndef _NO_UV
    #include <nbl/builtin/glsl/virtual_texturing/extensions.glsl>

    #define _NBL_VT_DESCRIPTOR_SET 0
    #define _NBL_VT_PAGE_TABLE_BINDING 0

    #define _NBL_VT_FLOAT_VIEWS_BINDING 1 
    #define _NBL_VT_FLOAT_VIEWS_COUNT 2
    #define _NBL_VT_FLOAT_VIEWS

    #define _NBL_VT_INT_VIEWS_BINDING 2
    #define _NBL_VT_INT_VIEWS_COUNT 0
    #define _NBL_VT_INT_VIEWS

    #define _NBL_VT_UINT_VIEWS_BINDING 3
    #define _NBL_VT_UINT_VIEWS_COUNT 0
    #define _NBL_VT_UINT_VIEWS
    #include <nbl/builtin/glsl/virtual_texturing/descriptors.glsl>

    layout (set = 2, binding = 0, std430) restrict readonly buffer PrecomputedStuffSSBO
    {
        uint pgtab_sz_log2;
        float vtex_sz_rcp;
        float phys_pg_tex_sz_rcp[_NBL_VT_MAX_PAGE_TABLE_LAYERS];
        uint layer_to_sampler_ix[_NBL_VT_MAX_PAGE_TABLE_LAYERS];
    } precomputed;
#endif
#define _NBL_FRAG_SET3_BINDINGS_DEFINED_

struct PCstruct
{
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    vec3 Ke;
    uvec2 map_Ka_data;
    uvec2 map_Kd_data;
    uvec2 map_Ks_data;
    uvec2 map_Ns_data;
    uvec2 map_d_data;
    uvec2 map_bump_data;
    float Ns;
    float d;
    float Ni;
    uint extra; //flags copied from MTL metadata
};
layout (push_constant) uniform Block {
    PCstruct params;
} PC;
#define _NBL_FRAG_PUSH_CONSTANTS_DEFINED_


#ifndef _NO_UV
    uint nbl_glsl_VT_layer2pid(in uint layer)
    {
        return precomputed.layer_to_sampler_ix[layer];
    }
    uint nbl_glsl_VT_getPgTabSzLog2()
    {
        return precomputed.pgtab_sz_log2;
    }
    float nbl_glsl_VT_getPhysPgTexSzRcp(in uint layer)
    {
        return precomputed.phys_pg_tex_sz_rcp[layer];
    }
    float nbl_glsl_VT_getVTexSzRcp()
    {
        return precomputed.vtex_sz_rcp;
    }
    #define _NBL_USER_PROVIDED_VIRTUAL_TEXTURING_FUNCTIONS_

    //nbl/builtin/glsl/virtual_texturing/functions.glsl/...
    #include <nbl/builtin/glsl/virtual_texturing/functions.glsl/7/8>
#endif


#ifndef _NO_UV
    vec4 nbl_sample_Ka(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Ka_data, uv, dUV); }

    vec4 nbl_sample_Kd(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Kd_data, uv, dUV); }

    vec4 nbl_sample_Ks(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Ks_data, uv, dUV); }

    vec4 nbl_sample_Ns(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_Ns_data, uv, dUV); }

    vec4 nbl_sample_d(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_d_data, uv, dUV); }

    vec4 nbl_sample_bump(in vec2 uv, in mat2 dUV) { return nbl_glsl_vTextureGrad(PC.params.map_bump_data, uv, dUV); }
#endif
#define _NBL_TEXTURE_SAMPLE_FUNCTIONS_DEFINED_


#include "nbl/builtin/shaders/loaders/mtl/fragment_impl.glsl"
