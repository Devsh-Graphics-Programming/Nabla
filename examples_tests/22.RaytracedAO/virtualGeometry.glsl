#ifndef _VIRTUAL_GEOMETRY_GLSL_INCLUDED_
#define _VIRTUAL_GEOMETRY_GLSL_INCLUDED_


#define _NBL_VG_USE_SSBO
#define _NBL_VG_SSBO_DESCRIPTOR_SET 1
#define _NBL_VG_USE_SSBO_UINT
#define _NBL_VG_SSBO_UINT_BINDING 0
#define _NBL_VG_USE_SSBO_UVEC3
#define _NBL_VG_SSBO_UVEC3_BINDING 1
#define _NBL_VG_USE_SSBO_INDEX
#define _NBL_VG_SSBO_INDEX_BINDING 2
// TODO: remove after all quantization optimizations in CSerializedLoader and the like
#define _NBL_VG_USE_SSBO_UVEC2
#define _NBL_VG_SSBO_UVEC2_BINDING 3
#include <nbl/builtin/glsl/virtual_geometry/virtual_attribute_fetch.glsl>


#include <nbl/builtin/glsl/ext/MitsubaLoader/instance_data_descriptor.glsl>


vec3 nbl_glsl_fetchVtxPos(in uint vtxID, in uint drawGUID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = InstData.data[drawGUID].padding0;
    return nbl_glsl_VG_attribFetch_RGB32_SFLOAT(va,vtxID);
}

vec3 nbl_glsl_fetchVtxNormal(in uint vtxID, in uint drawGUID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = InstData.data[drawGUID].padding1;
    return normalize(nbl_glsl_VG_attribFetch_RGB10A2_SNORM(va,vtxID).xyz);
}

vec2 nbl_glsl_fetchVtxUV(in uint vtxID, in uint drawGUID)
{
    nbl_glsl_VG_VirtualAttributePacked_t va = InstData.data[drawGUID].determinantSignBit;
    return nbl_glsl_VG_attribFetch_RG32_SFLOAT(va,vtxID);
}


#endif
