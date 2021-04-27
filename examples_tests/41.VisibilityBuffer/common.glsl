#ifndef _COMMON_GLSL_INCLUDED_
#define _COMMON_GLSL_INCLUDED_

#include "common.h"

// descriptors
#include <nbl/builtin/glsl/utils/common.glsl>
layout(set = 1, binding = 0, row_major, std140) uniform UBO
{
    nbl_glsl_SBasicViewParameters params;
} CamData;

#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute.glsl>
struct MeshBuffer_t
{
    nbl_glsl_VG_VirtualAttributePacked_t vAttr[USED_ATTRIBUTES];
    uint baseVertex;
};
layout(set = 2, binding = 0) readonly buffer VirtualAttributes
{
    nbl_glsl_VG_VirtualAttributePacked_t vAttr[][USED_ATTRIBUTES]; // TODO: replace
} virtualAttribTable;


// functions
#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute_fetch.glsl>
vec3 nbl_glsl_fetchVtxPos(in uint vtxID, in uint drawGUID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = virtualAttribTable.vAttr[drawGUID][0];
    return nbl_glsl_VG_attribFetch_RGB32_SFLOAT(va,vtxID);
}

vec2 nbl_glsl_fetchVtxUV(in uint vtxID, in uint drawGUID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = virtualAttribTable.vAttr[drawGUID][1];
    return nbl_glsl_VG_attribFetch_RG32_SFLOAT(va,vtxID);
}

vec3 nbl_glsl_fetchVtxNormal(in uint vtxID, in uint drawGUID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = virtualAttribTable.vAttr[drawGUID][2];
    return nbl_glsl_VG_attribFetch_RGB10A2_SNORM(va,vtxID).xyz;
}

#endif