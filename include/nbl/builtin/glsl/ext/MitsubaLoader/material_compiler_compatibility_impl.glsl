#ifndef _NBL_BUILTIN_GLSL_EXT_MITSUBA_LOADER_DESCRIPTORS_INCLUDED_
#define _NBL_BUILTIN_GLSL_EXT_MITSUBA_LOADER_DESCRIPTORS_INCLUDED_

#define _NBL_VT_DESCRIPTOR_SET 0
#define _NBL_VT_PAGE_TABLE_BINDING 0

#define _NBL_VT_FLOAT_VIEWS_BINDING 1 
#define _NBL_VT_FLOAT_VIEWS_COUNT _NBL_EXT_MITSUBA_LOADER_VT_STORAGE_VIEW_COUNT
#define _NBL_VT_FLOAT_VIEWS

#define _NBL_VT_INT_VIEWS_BINDING 2
#define _NBL_VT_INT_VIEWS_COUNT 0
#define _NBL_VT_INT_VIEWS

#define _NBL_VT_UINT_VIEWS_BINDING 3
#define _NBL_VT_UINT_VIEWS_COUNT 0
#define _NBL_VT_UINT_VIEWS
#include <nbl/builtin/glsl/virtual_texturing/descriptors.glsl>

layout(set = 0, binding = 2, std430) restrict readonly buffer VT_PrecomputedStuffSSBO
{
    uint pgtab_sz_log2;
    float vtex_sz_rcp;
    float phys_pg_tex_sz_rcp[_NBL_VT_MAX_PAGE_TABLE_LAYERS];
    uint layer_to_sampler_ix[_NBL_VT_MAX_PAGE_TABLE_LAYERS];
} VT_precomputed;

layout(set = 0, binding = 3, std430) restrict readonly buffer INSTR_BUF
{
    nbl_glsl_MC_instr_t data[];
} instr_buf;
layout(set = 0, binding = 4, std430) restrict readonly buffer BSDF_BUF
{
    nbl_glsl_MC_bsdf_data_t data[];
} bsdf_buf;
#include <nbl/builtin/glsl/ext/MitsubaLoader/instance_data_struct.glsl>
layout(set = 0, binding = 5, row_major, std430) readonly restrict buffer InstDataBuffer {
    nbl_glsl_ext_Mitsuba_Loader_instance_data_t data[];
} InstData;
layout(set = 0, binding = 6, std430) restrict readonly buffer PREFETCH_INSTR_BUF
{
    nbl_glsl_MC_prefetch_instr_t data[];
} prefetch_instr_buf;

// Note: this GLSL header defines just those 3 material compiler functions.
// The rest of them must still be defined by the user!
nbl_glsl_MC_instr_t nbl_glsl_MC_fetchInstr(in uint ix)
{
    return instr_buf.data[ix];
}
nbl_glsl_MC_prefetch_instr_t nbl_glsl_MC_fetchPrefetchInstr(in uint ix)
{
    return prefetch_instr_buf.data[ix];
}
nbl_glsl_MC_bsdf_data_t nbl_glsl_MC_fetchBSDFData(in uint ix)
{
    return bsdf_buf.data[ix];
}

uint nbl_glsl_VT_layer2pid(in uint layer)
{
    return VT_precomputed.layer_to_sampler_ix[layer];
}
uint nbl_glsl_VT_getPgTabSzLog2()
{
    return VT_precomputed.pgtab_sz_log2;
}
float nbl_glsl_VT_getPhysPgTexSzRcp(in uint layer)
{
    return VT_precomputed.phys_pg_tex_sz_rcp[layer];
}
float nbl_glsl_VT_getVTexSzRcp()
{
    return VT_precomputed.vtex_sz_rcp;
}
#define _NBL_USER_PROVIDED_VIRTUAL_TEXTURING_FUNCTIONS_

#include <nbl/builtin/glsl/virtual_texturing/functions.glsl/7/8>

#endif
