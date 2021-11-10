// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET_GLSL_INCLUDED_
#define _NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET_GLSL_INCLUDED_

#include <nbl/builtin/glsl/shapes/aabb.glsl>

#ifndef NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET
#define NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET 0
#endif

#ifndef NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_BINDING
#define NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_BINDING 0
#endif
#ifndef NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_QUALIFIERS
#define NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_QUALIFIERS readonly restrict
#endif
#ifndef NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_DECLARED
#define NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_DECLARED
layout(
    set=NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET,
    binding=NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_BINDING
) NBL_GLSL_LOD_LIBRARY_LOD_TABLES_DESCRIPTOR_QUALIFIERS buffer LodTables
{
    uvec4 data[];
} lodTables;

uint nbl_glsl_lod_library_Table_getLoDCount(in uint lodTableUvec4Offset)
{
    const uint offsetofLevelCount = 3u;
    return lodTables.data[lodTableUvec4Offset][offsetofLevelCount];
}
nbl_glsl_shapes_AABB_t nbl_glsl_lod_library_Table_getAABB(in uint lodTableUvec4Offset)
{
    const uint uvec4OffsetofMaxAABB = 1u;
    return nbl_glsl_shapes_AABB_t(
        uintBitsToFloat(lodTables.data[lodTableUvec4Offset].xyz),
        uintBitsToFloat(lodTables.data[lodTableUvec4Offset+uvec4OffsetofMaxAABB].xyz)
    );
}
uint nbl_glsl_lod_library_Table_getLoDUvec2Offset(in uint lodTableUvec4Offset, in uint lodID)
{
    const uint offsetofFirstLoDUvec2Offset = 7u;
    const uint offsetofLodUvec2Offset = lodID+offsetofFirstLoDUvec2Offset;
    return lodTables.data[lodTableUvec4Offset+(offsetofLodUvec2Offset>>2u)][offsetofLodUvec2Offset&0x3u];
}
#endif


#ifndef NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_BINDING
#define NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_BINDING 1
#endif
#ifndef NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_QUALIFIERS
#define NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_QUALIFIERS readonly restrict
#endif
#ifndef NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_DECLARED
#define NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_DECLARED
layout(
    set=NBL_GLSL_LOD_LIBRARY_DESCRIPTOR_SET,
    binding=NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_BINDING
) NBL_GLSL_LOD_LIBRARY_LOD_INFOS_DESCRIPTOR_QUALIFIERS buffer LodInfos
{
    uvec2 data[];
} lodInfos;

uint nbl_glsl_lod_library_Info_getDrawcallInfoCount(in uint lodInfoUvec2Offset)
{
    const int bitoffset_drawcallInfoCount = 0;
    return bitfieldExtract(lodInfos.data[lodInfoUvec2Offset][0],bitoffset_drawcallInfoCount,16);
}
uint nbl_glsl_lod_library_Info_getTotalBoneCount(in uint lodInfoUvec2Offset)
{
    const int bitoffset_totalBoneCount = 16;
    return bitfieldExtract(lodInfos.data[lodInfoUvec2Offset][0],bitoffset_totalBoneCount,16);
}

#include <nbl/builtin/glsl/lod_library/structs.glsl>
#include <nbl/builtin/glsl/format/decode.glsl>
nbl_glsl_shapes_AABB_t nbl_glsl_lod_library_Info_getAABB(in uint lodInfoUvec2Offset, in uint offsetofUvec2FirstDrawcallInfo, in uint drawcallID)
{
    const uint uvec2OffsetofMinAABB = 0u;
    const uint uvec2OffsetofMaxAABB = 1u;
    const uint offset = lodInfoUvec2Offset+offsetofUvec2FirstDrawcallInfo+drawcallID*NBL_GLSL_LOD_LIBRARY_DRAWCALL_INFO_UVEC2_SIZE;
    return nbl_glsl_shapes_AABB_t(
        nbl_glsl_decodeRGB18E7S3(lodInfos.data[offset+uvec2OffsetofMinAABB]),
        nbl_glsl_decodeRGB18E7S3(lodInfos.data[offset+uvec2OffsetofMaxAABB])
    );
}
uint nbl_glsl_lod_library_Info_getDrawCallDWORDOffset(in uint lodInfoUvec2Offset, in uint offsetofUvec2FirstDrawcallInfo, in uint drawcallID)
{
    const uint uvec2OffsetofDrawcallDWORDOffset = 2u;
    const uint offset = lodInfoUvec2Offset+offsetofUvec2FirstDrawcallInfo+drawcallID*NBL_GLSL_LOD_LIBRARY_DRAWCALL_INFO_UVEC2_SIZE;
    return lodInfos.data[offset+uvec2OffsetofDrawcallDWORDOffset][0];
}


nbl_glsl_lod_library_DefaultLoDChoiceParams nbl_glsl_lod_library_DefaultInfo_getLoDChoiceParams(in uint lodInfoUvec2Offset)
{
    const uint offsetofUvec2_lodChoiceParams = NBL_GLSL_LOD_LIBRARY_LOD_INFO_BASE_SIZE>>3u;
    return nbl_glsl_lod_library_DefaultLoDChoiceParams(uintBitsToFloat(
        lodInfos.data[lodInfoUvec2Offset+offsetofUvec2_lodChoiceParams][(NBL_GLSL_LOD_LIBRARY_LOD_INFO_BASE_SIZE>>2)&0x1u]
    ));
}

nbl_glsl_shapes_AABB_t nbl_glsl_lod_library_DefaultInfo_getAABB(in uint lodInfoUvec2Offset, in uint drawcallID)
{
    return nbl_glsl_lod_library_Info_getAABB(
        lodInfoUvec2Offset,NBL_GLSL_CULLING_LOD_SELECTION_LOD_INFO_DRAWCALL_LIST_UVEC2_OFFSET,drawcallID
    );
}
uint nbl_glsl_lod_library_DefaultInfo_getDrawCallDWORDOffset(in uint lodInfoUvec2Offset, in uint drawcallID)
{
    return nbl_glsl_lod_library_Info_getDrawCallDWORDOffset(
        lodInfoUvec2Offset,NBL_GLSL_CULLING_LOD_SELECTION_LOD_INFO_DRAWCALL_LIST_UVEC2_OFFSET,drawcallID
    );
}
#endif


#endif