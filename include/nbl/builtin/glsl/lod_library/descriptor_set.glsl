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
    return nbl_glsl_shapes_AABB_t(uintBitsToFloat(lodTables.data[lodTableUvec4Offset].xyz),uintBitsToFloat(lodTables.data[lodTableUvec4Offset+uvec4OffsetofMaxAABB].xyz));
}
uint nbl_glsl_lod_library_Table_getLoDUvec4Offset(in uint lodTableUvec4Offset, in uint lodID)
{
    const uint offsetofFirstLoDUvec4Offset = 7u;
    const uint offsetofLodUvec4Offset = lodID+offsetofFirstLoDUvec4Offset;
    return lodTables.data[lodTableUvec4Offset+(offsetofLodUvec4Offset>>2u)][offsetofLodUvec4Offset&0x3u];
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
    uint data[];
} lodInfos;

uint nbl_glsl_lod_library_Info_getDrawcallInfoCount(in uint lodInfoUvec4Offset)
{
    const uint lodInfoDWORDOffset = lodInfoUvec4Offset<<2u;
    const uint offsetof_drawcallInfoCountAndTotalBoneCount = 3u;
    return lodInfos.data[lodInfoDWORDOffset+offsetof_drawcallInfoCountAndTotalBoneCount]&0xffffu;
}
uint nbl_glsl_lod_library_Info_getTotalBoneCount(in uint lodInfoUvec4Offset)
{
    const uint lodInfoDWORDOffset = lodInfoUvec4Offset<<2u;
    const uint offsetof_drawcallInfoCountAndTotalBoneCount = 3u;
    return lodInfos.data[lodInfoDWORDOffset+offsetof_drawcallInfoCountAndTotalBoneCount]>>16u;
}

nbl_glsl_shapes_AABB_t nbl_glsl_lod_library_Info_getAABB(in uint lodInfoUvec4Offset)
{
    const uint lodInfoDWORDOffset = lodInfoUvec4Offset<<2u;
    const uint offsetof_aabbMax = 4u;
    return nbl_glsl_shapes_AABB_t(
        vec3(
            uintBitsToFloat(lodInfos.data[lodInfoDWORDOffset+0u]),
            uintBitsToFloat(lodInfos.data[lodInfoDWORDOffset+1u]),
            uintBitsToFloat(lodInfos.data[lodInfoDWORDOffset+2u])
        ),
        vec3(
            uintBitsToFloat(lodInfos.data[lodInfoDWORDOffset+offsetof_aabbMax+0u]),
            uintBitsToFloat(lodInfos.data[lodInfoDWORDOffset+offsetof_aabbMax+1u]),
            uintBitsToFloat(lodInfos.data[lodInfoDWORDOffset+offsetof_aabbMax+2u])
        )
    );
}

uint nbl_glsl_lod_library_Info_getDrawCallDWORDOffset(in uint lodInfoUvec4Offset, in uint offsetofDWORDFirstDrawcallInfo, in uint drawcallID)
{
    const uint lodInfoDWORDOffset = lodInfoUvec4Offset<<2u;
    return lodInfos.data[lodInfoDWORDOffset+offsetofDWORDFirstDrawcallInfo+(drawcallID<<1u)];
}


#include <nbl/builtin/glsl/lod_library/structs.glsl>
nbl_glsl_lod_library_DefaultLoDChoiceParams nbl_glsl_lod_library_DefaultInfo_getLoDChoiceParams(in uint lodInfoUvec4Offset)
{
    const uint lodInfoDWORDOffset = lodInfoUvec4Offset<<2u;
    const uint offsetof_lodChoiceParams = NBL_GLSL_LOD_LIBRARY_LOD_INFO_BASE_SIZE>>2u;
    return nbl_glsl_lod_library_DefaultLoDChoiceParams(uintBitsToFloat(lodInfos.data[lodInfoDWORDOffset+offsetof_lodChoiceParams]));
}

uint nbl_glsl_lod_library_DefaultInfo_getDrawCallDWORDOffset(in uint lodInfoUvec4Offset, in uint drawcallID)
{
    return nbl_glsl_lod_library_Info_getDrawCallDWORDOffset(
        lodInfoUvec4Offset,NBL_GLSL_CULLING_LOD_SELECTION_LOD_INFO_DRAWCALL_LIST_DWORD_OFFSET,drawcallID
    );
}
#endif


#endif