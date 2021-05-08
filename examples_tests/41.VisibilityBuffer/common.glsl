#ifndef _COMMON_GLSL_INCLUDED_
#define _COMMON_GLSL_INCLUDED_

#extension GL_EXT_shader_16bit_storage: require

#include "common.h"

// defines for buffer fill pipeline
#define NBL_GLSL_BARYCENTRIC_VERT_POS_OUTPUT_LOC 1
#define NBL_GLSL_BARYCENTRIC_VERT_PROVOKINGPOS_OUTPUT_LOC 2
#define NBL_GLSL_BARYCENTRIC_FRAG_POS_INPUT_LOC NBL_GLSL_BARYCENTRIC_VERT_POS_OUTPUT_LOC
#define NBL_GLSL_BARYCENTRIC_FRAG_PROVOKINGPOS_INPUT_LOC NBL_GLSL_BARYCENTRIC_VERT_PROVOKINGPOS_OUTPUT_LOC

#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute.glsl>
struct BatchInstanceData
{
    vec3 Ka;
    uint firstIndex;
    vec3 Kd;
    nbl_glsl_VG_VirtualAttributePacked_t vAttrPos;
    vec3 Ks;
    nbl_glsl_VG_VirtualAttributePacked_t vAttrUV;
    vec3 Ke;
    nbl_glsl_VG_VirtualAttributePacked_t vAttrNormal;
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

layout(set = 1, binding = 0, std430) readonly buffer BatchInstanceBuffer
{
    BatchInstanceData batchInstanceData[];
};

// non-global descriptors
#include <nbl/builtin/glsl/utils/common.glsl>
layout(set = 2, binding = 0, row_major, std140) uniform UBO
{
    nbl_glsl_SBasicViewParameters params;
} CamData;

// functions
#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute_fetch.glsl>
vec3 nbl_glsl_fetchVtxPos(in uint vtxID, in uint drawGUID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = batchInstanceData[drawGUID].vAttrPos;
    return nbl_glsl_VG_attribFetch_RGB32_SFLOAT(va,vtxID);
}

vec2 nbl_glsl_fetchVtxUV(in uint vtxID, in uint drawGUID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = batchInstanceData[drawGUID].vAttrUV;
    return nbl_glsl_VG_attribFetch_RG32_SFLOAT(va,vtxID);
}

vec3 nbl_glsl_fetchVtxNormal(in uint vtxID, in uint drawGUID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = batchInstanceData[drawGUID].vAttrNormal;
    return normalize(nbl_glsl_VG_attribFetch_RGB10A2_SNORM(va,vtxID).xyz);
}
#endif