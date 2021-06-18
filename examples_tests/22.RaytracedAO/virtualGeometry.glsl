#ifndef _VIRTUAL_GEOMETRY_GLSL_INCLUDED_
#define _VIRTUAL_GEOMETRY_GLSL_INCLUDED_

#include "common.h"

#define _NBL_VG_USE_SSBO
#define _NBL_VG_SSBO_DESCRIPTOR_SET 1
#define _NBL_VG_USE_SSBO_UVEC2
#define _NBL_VG_SSBO_UVEC2_BINDING 0
#define _NBL_VG_USE_SSBO_INDEX
#define _NBL_VG_SSBO_INDEX_BINDING 1
// TODO: remove after Doom Eternal position quantization trick
#define _NBL_VG_USE_SSBO_UVEC3
#define _NBL_VG_SSBO_UVEC3_BINDING 2
#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute_fetch.glsl>


#include <nbl/builtin/glsl/ext/MitsubaLoader/instance_data_descriptor.glsl>


vec3 nbl_glsl_fetchVtxPos(in uint vtxID, in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = batchInstanceData.padding1;
    return nbl_glsl_VG_attribFetch_RGB32_SFLOAT(va,vtxID);
}

vec3 nbl_glsl_fetchVtxNormal(in uint vtxID, in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = batchInstanceData.determinantSignBit;
    const uint codedNormal = nbl_glsl_VG_attribFetch2u(va,vtxID)[0];
    return normalize(nbl_glsl_decodeRGB10A2_SNORM(codedNormal).xyz);
}

vec2 nbl_glsl_fetchVtxUV(in uint vtxID, in nbl_glsl_ext_Mitsuba_Loader_instance_data_t batchInstanceData)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = batchInstanceData.determinantSignBit;
    const uint codedUV = nbl_glsl_VG_attribFetch2u(va,vtxID)[1];
    return unpackHalf2x16(codedUV).xy;
}


#endif
